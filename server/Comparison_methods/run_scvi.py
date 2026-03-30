import time
import os
import h5py
import numpy as np
import scanpy as sc
import scvi
from reproducibility.utils import data_sample, set_seed, calculate_metric, read_data

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
adata = sc.AnnData(x)
adata.obs['cell_type'] = y
adata.obs['cell_type'] = adata.obs['cell_type'].astype(str).astype('category')

adata.raw = adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["counts"] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts",n_top_genes=2000, subset=True)

set_seed(0)
start = time.time()

scvi.model.SCVI.setup_anndata(adata,layer="counts")
vae = scvi.model.SCVI(adata)
vae.train()
adata.obsm["X_scVI"] = vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.leiden(adata)
pred = np.array(adata.obs['leiden'])

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

embedding = np.array(adata.obsm["X_scVI"])
k = len(np.unique(pred))
nmi, ari = calculate_metric(pred, np.array(adata.obs['cell_type']).squeeze())

print("ARI: ", ari)
print("NMI:", nmi)
print("k:", k)

base = os.path.splitext(os.path.basename(args.input_file))[0]
out_file = os.path.join(out_dir, f"results_{base}.npz")
np.savez(out_file, ARI=ari, NMI=nmi, K=k, Clusters=pred, Embedding=embedding, Time_use=run_time)