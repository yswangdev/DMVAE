import numpy as np
import scanpy as sc
import os
from reproducibility.utils import set_seed
from scace import run_scace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True, help="Path to input data directory")
parser.add_argument("--out-dir", type=str, required=True, help="Path for output results")
parser.add_argument("--expression-file", type=str, required=True, help="Expression file (txt/csv dense matrix)")
parser.add_argument("--labels-file", type=str, required=True, help="Label file (txt/csv vector)")
args = parser.parse_args()

input_dir = args.data_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

x_path = os.path.join(input_dir, args.expression_file)
y_path = os.path.join(input_dir, args.labels_file)

if not os.path.isfile(x_path):
    raise FileNotFoundError(f"Expression not found: {x_path}")
if not os.path.isfile(y_path):
    raise FileNotFoundError(f"Labels not found: {y_path}")

print(f"[RUN ] Loading {x_path} and {y_path}")

# Load data
x = np.loadtxt(x_path, delimiter="," if x_path.lower().endswith(".csv") else None)
# If rows >> cols, assume genes x cells -> transpose to cells x genes
if x.shape[0] > x.shape[1]:
    x = x.T

y = np.loadtxt(y_path, delimiter="," if y_path.lower().endswith(".csv") else None).astype(int).reshape(-1)


# Build AnnData
adata = sc.AnnData(x)

# Check label length
if len(y) != adata.n_obs:
    raise ValueError(f"Label length ({len(y)}) != number of cells ({adata.n_obs}).")

adata.obs["celltype"] = y

# Reproducibility seed
set_seed(42)

# ---- Basic QC ----
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)

# Keep raw
adata.raw = adata.copy()

# ---- Normalize, log, scale ----
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# ---- Run SCACE ----
adata, nmi, ari, K, pred_all, emb_all, run_time = run_scace(
    adata, cl_type="celltype", return_all=True
)

# ---- Save results ----
base = os.path.splitext(os.path.basename(args.expression_file))[0]
out_file = os.path.join(out_dir, f"results_{base}.npz")
np.savez(
    out_file,
    ARI=np.array(ari, dtype=object),
    NMI=np.array(nmi, dtype=object),
    K=np.array(K, dtype=object),
    Embedding=np.array(emb_all, dtype=object),
    Clusters=np.array(pred_all, dtype=object),
    Labels=y,
    Time=np.array(run_time, dtype=object)
)
print(f"[SAVE] Dataset {base}: {out_file}")