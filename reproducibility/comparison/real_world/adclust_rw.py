import argparse
import numpy as np
import scanpy as sc
from ADClust import ADClust
import time
import os
import time
from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data

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

print(f"[RUN ] Loading {x_path} and {y_path}")

# Load data
x = np.loadtxt(x_path, delimiter="," if x_path.lower().endswith(".csv") else None)
if x.shape[0] > x.shape[1]:
    x = x.T
y = np.loadtxt(y_path, delimiter="," if y_path.lower().endswith(".csv") else None).astype(int).reshape(-1)

# Build AnnData
adata = sc.AnnData(x)
adata.obs["CellType"] = y.astype(str)
adata.obs["CellType"] = adata.obs["CellType"].astype("category")

adata = data_preprocess(adata, select_gene_adclust=True)
data = adata.X
labels = adata.obs['CellType'].values.astype(np.int32)

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

base = os.path.splitext(os.path.basename(args.expression_file))[0]
out_file = os.path.join(out_dir, f"results_{base}.npz")
np.savez(out_file, ARI=ari, NMI=nmi, K=K, Embedding=embedded,
         Clusters=cluster_labels, Clusters_merge=pred, Clusters_merge_all=pred_all, Time_use=run_time)