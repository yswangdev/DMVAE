import time
import os
import h5py
import numpy as np
import scanpy as sc
import scvi
from reproducibility.utils import data_sample, set_seed, calculate_metric, read_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_datafile", type=str, required=True,
                    help="Path to input data directory")
parser.add_argument("--output_path", type=str, required=True,
                    help="Path for output results")
parser.add_argument("--start", type=int, default=1,
                    help="Start index for data (inclusive)")
parser.add_argument("--end", type=int, default=10,
                    help="End index for data (exclusive, like range)")
args = parser.parse_args()

input_dir = args.input_datafile
out_dir = args.output_path
os.makedirs(out_dir, exist_ok=True)


def run_one(idx: int):
    x_path = os.path.join(input_dir, f"simdata_{idx}.txt")
    y_path = os.path.join(input_dir, f"simmeta_{idx}.txt")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        print(f"[SKIP] {idx}: missing file(s): {x_path if not os.path.isfile(x_path) else ''} {y_path if not os.path.isfile(y_path) else ''}")
        return

    print(f"[RUN ] Dataset {idx}: loading {x_path} and {y_path}")

    # Load data
    x = np.loadtxt(x_path)
    y = np.loadtxt(y_path).astype(int)

    # Build AnnData
    adata = sc.AnnData(x)
    adata.obs['cell_type'] = y
    adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).astype('category')

    # -------------------- Preprocess for scvi --------------------
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts",n_top_genes=2000, subset=True)


    # -------------------- Set seed (vary by idx if desired) --------------------
    set_seed(42 + idx)

     # -------------------- Run scVI -------------------
    start_t = time.time()

    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    vae = scvi.model.SCVI(adata)
    vae.train()

    adata.obsm["X_scVI"] = vae.get_latent_representation()

    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.leiden(adata)
    pred = np.array(adata.obs["leiden"])

    end_t = time.time()
    run_time = end_t - start_t
    print(f"[TIME] Dataset {idx}: {run_time:.3f} seconds")

    # -------------------- Metrics --------------------
    embedding = np.array(adata.obsm["X_scVI"])
    k = len(np.unique(pred))
    labels = np.array(adata.obs["cell_type"]).squeeze()

    nmi, ari = calculate_metric(pred, labels)

    print(f"[METRIC] Dataset {idx}: ARI={ari:.4f}, NMI={nmi:.4f}, k={k}")

    # -------------------- Save results --------------------
    out_file = os.path.join(out_dir, f"results_sim_{idx}.npz")
    np.savez(
        out_file,
        ARI=ari,
        NMI=nmi,
        K=k,
        Clusters=pred,
        Embedding=embedding,
        Time_use=run_time,
    )
    print(f"[SAVE] Dataset {idx}: {out_file}")


# -------------------- Batch runner --------------------
for i in range(args.start, args.end):
    try:
        run_one(i)
    except Exception as e:
        print(f"[FAIL] Dataset {i}: {e}")