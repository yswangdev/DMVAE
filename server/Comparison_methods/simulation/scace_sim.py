import numpy as np
import scanpy as sc
import os
from reproducibility.utils import data_sample, data_preprocess, set_seed
from scace import run_scace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_datafile", type=str, required=True, help="Path to input data directory")
parser.add_argument("--output_path", type=str, required=True, help="Path for output results")
parser.add_argument("--start", type=int, default=1, help="Start index for data")
parser.add_argument("--end", type=int, default=10, help="End index for data")
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
    adata.obs["celltype"] = y

    # Reproducibility seed (vary by idx to avoid identical randomness if desired)
    set_seed(42 + idx)

    # ---- Basic QC ----
    # Filter genes/cells
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)

    # Keep raw
    adata.raw = adata.copy()

    # ---- Normalize, log, scale ----
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # ---- Run SCACE ----
    adata, nmi, ari, K, pred_all, emb_all, _ = run_scace(
        adata, cl_type="celltype", return_all=True
    )

    # ---- Save results ----
    out_file = os.path.join(out_dir, f"results_sim_{idx}.npz")
    np.savez(out_file,
            ARI=np.array(ari, dtype=object),
            NMI=np.array(nmi, dtype=object),
            K=np.array(K, dtype=object),
            Embedding=np.array(emb_all, dtype=object),
            Clusters=np.array(pred_all, dtype=object),
            Labels=np.array(adata.obs["celltype"]).astype(int))
    print(f"[SAVE] Dataset {idx}: {out_file}")
    

for i in range(args.start, args.end):
    try:
        run_one(i)
    except Exception as e:
        print(f"[FAIL] Dataset {i}: {e}")
        
        

