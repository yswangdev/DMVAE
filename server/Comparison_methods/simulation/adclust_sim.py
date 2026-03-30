import argparse
import numpy as np
import scanpy as sc
import os
import time

from ADClust import ADClust
from reproducibility.utils import (
    data_sample,
    data_preprocess,
    set_seed,
    calculate_metric,
    read_data,
)

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
    adata.obs["CellType"] = y
    adata.obs["CellType"] = adata.obs["CellType"].astype("category")

    # -------------------- Preprocess for ADClust --------------------
    # (your reproducibility pipeline)
    adata = data_preprocess(adata, select_gene_adclust=True)
    data = adata.X
    labels = adata.obs["CellType"].values.astype(np.int32)

    # -------------------- Set seed (vary by idx if desired) --------------------
    set_seed(42 + idx)

    # -------------------- Run ADClust --------------------
    adClust = ADClust(data_size=data.shape[0])

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
    end_t = time.time()
    run_time = end_t - start_t

    print(f"[INFO] Dataset {idx}: estimated K = {estimated_cluster_numbers}")
    print(f"[TIME] Dataset {idx}: {run_time:.3f} seconds")

    # -------------------- Metrics --------------------
    nmi, ari = calculate_metric(labels, cluster_labels)
    K = estimated_cluster_numbers
    print(f"[METRIC] Dataset {idx}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # -------------------- Save results --------------------
    out_file = os.path.join(out_dir, f"results_sim_{idx}.npz")
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
    )
    print(f"[SAVE] Dataset {idx}: {out_file}")


# -------------------- Batch runner --------------------
for i in range(args.start, args.end):
    try:
        run_one(i)
    except Exception as e:
        print(f"[FAIL] Dataset {i}: {e}")