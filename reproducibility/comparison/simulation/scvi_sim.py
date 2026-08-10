"""Run scVI on simulated data, matching the scAce-repo github pipeline.

Input (``--input-format``, default ``txt``): ``simdata_{idx}.txt`` (cells x genes,
LOG-NORMALISED) + ``simmeta_{idx}.txt`` (integer labels). Pass ``h5`` to read the raw-count
``sim_{idx}.h5`` instead, which is the interface the scAce github pipeline uses.

CAVEAT for txt input: scVI's negative-binomial likelihood and scanpy's ``seurat_v3`` HVG
selection both expect RAW COUNTS. On log-normalised input both still run but warn, and the
likelihood is misspecified. Use ``--input-format h5`` for the statistically correct setup.
Preprocessing is verbatim from the github run_scvi.py: normalize_total(1e4) ->
log1p -> counts layer -> seurat_v3 HVG (2000) -> SCVI -> Leiden. The only change
vs the prior local version is reading RAW COUNTS (so normalization is applied
once, from counts, as the github intends).
"""

import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scvi

from reproducibility.utils import set_seed, calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True, help="Path to input data directory")
parser.add_argument("--out-dir", type=str, required=True, help="Path for output results")
parser.add_argument("--rep-start", type=int, default=1, help="Start index for data (inclusive)")
parser.add_argument("--rep-end", type=int, default=10, help="End index for data (exclusive)")
parser.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="Leiden resolution (scanpy default 1.0). Higher -> more clusters.",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Suffix for output filenames, e.g. '__res0.8', so parameter sweeps do not overwrite each other.",
)
parser.add_argument(
    "--input-format",
    choices=("txt", "h5"),
    default="txt",
    help="txt (default): simdata_{i}.txt + simmeta_{i}.txt (log-normalised counts x genes). "
    "h5: sim_{i}.h5 raw counts, as the scAce github pipeline uses.",
)
args = parser.parse_args()

input_dir = args.data_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)


def _finalise(x, y, src):
    """Coerce (X, y) to cells-x-genes float32 + contiguous int labels, or None."""
    x = np.asarray(x)
    if y is None:
        print(f"[FAIL] {src}: no 'Y' labels found")
        return None
    y = np.asarray(y).ravel()
    if y.dtype.kind in "SUO":                     # cell-type names, bytes or str
        y = np.array([v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in y])
        y = np.unique(y, return_inverse=True)[1]
    y = y.astype(int)
    # h5 written from R arrives transposed; orient rows = cells using the label count.
    if x.ndim == 2 and x.shape[0] != len(y) and x.shape[1] == len(y):
        print(f"[INFO] {os.path.basename(src)}: transposing X {x.shape} -> cells x genes")
        x = x.T
    if x.shape[0] != len(y):
        print(f"[FAIL] {src}: X {x.shape} does not match {len(y)} labels")
        return None
    return x.astype(np.float32), y


def _load_input(idx: int):
    """Return (X, y) for replicate ``idx``, or None if the input is absent.

    ``--data-dir`` may be either:

    * a single ``.h5`` FILE -- for real datasets such as ``mouse_h/data.h5`` that are
      not split into replicates. ``idx`` is ignored; the file is read directly,
      expecting datasets ``X`` (expression) and ``Y`` (labels).

    * a DIRECTORY of per-replicate simulation files. With ``--input-format txt`` (the
      default) it reads ``simdata_{idx}.txt`` (cells x genes, LOG-NORMALISED by the R
      generator) + ``simmeta_{idx}.txt``; with ``--input-format h5`` it reads
      ``sim_{idx}.h5`` (X: cells x genes RAW COUNTS, Y: labels), the interface the
      scAce github pipeline uses.

    NOTE (txt input): simdata is ALREADY log-normalised, so the preprocessing below
    normalises and log1p's a second time. Pass a raw-count .h5 to avoid that.
    """
    # ---- a single .h5 file: real dataset, no replicates ----
    if os.path.isfile(input_dir) and input_dir.endswith((".h5", ".hdf5")):
        print(f"[RUN ] Dataset {idx}: loading {input_dir}")
        with h5py.File(input_dir, "r") as f:
            if "X" not in f:
                print(f"[FAIL] {input_dir}: no 'X' dataset (found: {list(f.keys())})")
                return None
            x = np.array(f["X"])
            y = np.array(f["Y"]) if "Y" in f else None
        return _finalise(x, y, input_dir)

    if not os.path.isdir(input_dir):
        print(f"[FAIL] --data-dir is neither a .h5 file nor a directory: {input_dir}")
        return None

    # ---- a directory of per-replicate simulation files ----
    if args.input_format == "txt":
        dpath = os.path.join(input_dir, f"simdata_{idx}.txt")
        mpath = os.path.join(input_dir, f"simmeta_{idx}.txt")
        if not (os.path.isfile(dpath) and os.path.isfile(mpath)):
            print(f"[SKIP] {idx}: missing {dpath} or {mpath}")
            return None
        print(f"[RUN ] Dataset {idx}: loading {dpath} + {mpath}")
        x = pd.read_csv(dpath, sep=r"\s+", header=None).to_numpy(dtype=np.float32)
        y = pd.read_csv(mpath, sep=r"\s+", header=None).to_numpy().ravel().astype(int)
        return _finalise(x, y, dpath)

    h5_path = os.path.join(input_dir, f"sim_{idx}.h5")
    if not os.path.isfile(h5_path):
        print(f"[SKIP] {idx}: missing {h5_path}")
        return None
    print(f"[RUN ] Dataset {idx}: loading {h5_path}")
    with h5py.File(h5_path, "r") as f:
        return _finalise(np.array(f["X"]), np.array(f["Y"]), h5_path)


def run_one(idx: int):
    loaded = _load_input(idx)
    if loaded is None:
        return
    x, y = loaded

    adata = sc.AnnData(x.astype(np.float32))
    adata.obs["cell_type"] = y
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).astype("category")

    # ---- Preprocess for scVI (verbatim github run_scvi.py) ----
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=2000, subset=True
    )

    set_seed(42 + idx)

    # ---- Run scVI ----
    start_t = time.time()
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    vae = scvi.model.SCVI(adata)
    vae.train()
    adata.obsm["X_scVI"] = vae.get_latent_representation()

    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.leiden(adata, resolution=args.resolution)
    pred = np.array(adata.obs["leiden"])
    run_time = time.time() - start_t

    embedding = np.array(adata.obsm["X_scVI"])
    k = len(np.unique(pred))
    labels = np.array(adata.obs["cell_type"]).squeeze()
    nmi, ari = calculate_metric(pred, labels)
    print(f"[METRIC] Dataset {idx}: resolution={args.resolution}, ARI={ari:.4f}, NMI={nmi:.4f}, k={k}")

    # Leiden labels are strings, which np.savez can only store pickled. Also store
    # integer codes, assigned by sorted string order -- ids differ from the Leiden
    # names but the partition is identical.
    pred_int = np.unique(pred, return_inverse=True)[1].astype(np.int32)

    out_file = os.path.join(out_dir, f"results_sim_{idx}{args.tag}.npz")
    np.savez(
        out_file,
        ARI=ari,
        NMI=nmi,
        K=k,
        Clusters=pred,
        Clusters_int=pred_int,
        Embedding=embedding,
        Labels=labels.astype(int),
        Resolution=args.resolution,
        Time_use=run_time,
    )
    print(f"[SAVE] Dataset {idx}: {out_file}")


for i in range(args.rep_start, args.rep_end):
    try:
        run_one(i)
    except Exception as e:
        print(f"[FAIL] Dataset {i}: {e}")
