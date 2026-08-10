"""Run ADClust on simulated data, matching the scAce-repo github pipeline.

Input (``--input-format``, default ``txt``): ``simdata_{idx}.txt`` (cells x genes,
LOG-NORMALISED) + ``simmeta_{idx}.txt`` (integer labels). Pass ``h5`` to read the raw-count
``sim_{idx}.h5``, the interface run_adclust.py uses.
Preprocessing uses ``data_preprocess(adata, select_gene_adclust=True)``
(filter -> normalize_per_cell -> log1p -> HVG n_top_genes=2000 -> scale),
exactly as the github run script.
"""

import argparse
import os
import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from ADClust import ADClust
from reproducibility.utils import data_preprocess, set_seed, calculate_metric

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True, help="Path to input data directory")
parser.add_argument("--out-dir", type=str, required=True, help="Path for output results")
parser.add_argument("--rep-start", type=int, default=1, help="Start index for data (inclusive)")
parser.add_argument("--rep-end", type=int, default=10, help="End index for data (exclusive)")
parser.add_argument(
    "--dip-merge-threshold",
    type=float,
    default=0.9,
    help="ADClust Dip-test merge threshold (ADClust default 0.9). Micro-clusters merge "
    "while max Dip-score >= threshold, so HIGHER -> less merging -> MORE clusters.",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Suffix for output filenames, e.g. '__dip0.95', so parameter sweeps do not overwrite each other.",
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

    # float so in-place normalize_per_cell (in data_preprocess) can divide
    adata = sc.AnnData(x.astype(np.float32))
    adata.obs["CellType"] = y
    adata.obs["CellType"] = adata.obs["CellType"].astype("category")

    # ---- ADClust preprocessing (github run_adclust.py) ----
    adata = data_preprocess(adata, select_gene_adclust=True)
    data = adata.X
    labels = adata.obs["CellType"].values.astype(np.int32)

    set_seed(42 + idx)

    # ---- Run ADClust ----
    adClust = ADClust(data_size=data.shape[0], dip_merge_threshold=args.dip_merge_threshold)
    start_t = time.time()
    (
        cluster_labels,
        estimated_cluster_numbers,
        pred_all,
        pred,
        embedded,
        tsne,
        clusters,
    ) = adClust.fit(data)
    run_time = time.time() - start_t

    print(f"[INFO] Dataset {idx}: estimated k = {estimated_cluster_numbers}")
    nmi, ari = calculate_metric(labels, cluster_labels)
    K = estimated_cluster_numbers
    print(f"[METRIC] Dataset {idx}: dip_merge_threshold={args.dip_merge_threshold}, ARI={ari:.4f}, NMI={nmi:.4f}")

    out_file = os.path.join(out_dir, f"results_sim_{idx}{args.tag}.npz")
    np.savez(
        out_file,
        ARI=ari,
        NMI=nmi,
        K=K,
        Embedding=embedded,
        Clusters=cluster_labels,
        Clusters_merge=pred,
        Clusters_merge_all=pred_all,
        Time_use=run_time,
        Labels=labels,
        DipMergeThreshold=args.dip_merge_threshold,
    )
    print(f"[SAVE] Dataset {idx}: {out_file}")


for i in range(args.rep_start, args.rep_end):
    try:
        run_one(i)
    except Exception as e:
        print(f"[FAIL] Dataset {i}: {e}")
