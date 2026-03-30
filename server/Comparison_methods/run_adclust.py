import h5py
import numpy as np
import scanpy as sc
from ADClust import ADClust
import time
import os
import time
from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data
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
adata.obs['CellType'] = y
adata.obs['CellType'] = adata.obs['CellType'].astype(str).astype('category')

adata = data_preprocess(adata, select_gene_adclust=True)
data = adata.X
labels = adata.obs['CellType'].astype('category').cat.codes.values.astype(np.int32)

set_seed(0)
adClust = ADClust(data_size=data.shape[0])

start = time.time()
cluster_labels, estimated_cluster_numbers, pred_all, pred, embedded, tsne, clusters = adClust.fit(data)
print("The estimated number of clusters:", estimated_cluster_numbers)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

nmi, ari = calculate_metric(labels, cluster_labels)
K = estimated_cluster_numbers
print("ARI: ", ari)
print("NMI:", nmi)

base = os.path.splitext(os.path.basename(args.input_file))[0]
out_file = os.path.join(out_dir, f"results_{base}.npz")
np.savez(out_file, ARI=ari, NMI=nmi, K=K, Embedding=embedded,
         Clusters=cluster_labels, Clusters_merge=pred, Clusters_merge_all=pred_all, Time_use=run_time)