import h5py
import numpy as np
import scanpy as sc
import os

from reproducibility.utils import data_sample, data_preprocess, set_seed, read_data
from scace import run_scace
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_datafile", type=str, required=True, help="Path to input data directory")
parser.add_argument("--output_path", type=str, required=True, help="Path for output results")
parser.add_argument("--input_file", type=str, required=True, help="Expression file (txt/csv dense matrix)")
args = parser.parse_args()

input_dir = args.input_datafile
out_dir = args.output_path
os.makedirs(out_dir, exist_ok=True)

####################################  Read dataset  ####################################
x_path = os.path.join(input_dir, args.input_file)
dataset_name = os.path.basename(os.path.normpath(input_dir))

if dataset_name in {"human_k", "human_p"}:
    with h5py.File(x_path, "r") as data_mat:
        x = np.array(data_mat["X"])
        y = np.array(data_mat["Y"])
else:
    mat, obs, var, uns = read_data(x_path, sparsify=False, skip_exprs=False)
    x = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)
adata, nmi, ari, K, pred_all, emb_all, run_time = run_scace(adata, cl_type='celltype', return_all=True)
base = os.path.splitext(os.path.basename(args.input_file))[0]
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