import time
import os
import desc as desc
import h5py
import numpy as np
import scanpy as sc

from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data


sc.settings.verbosity = 3
sc.logging.print_versions()

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
mat, obs, var, uns = read_data(x_path, sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Run without sampling  ####################################

adata = sc.AnnData(x)
adata.obs['celltype'] = y
adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype('category')

adata = data_preprocess(adata, scale_factor=False, counts_per_cell=True,
                        normalize_input=False, select_gene_desc=True)
desc.scale(adata, zero_center=True, max_value=3)


start = time.time()
set_seed(0)
adata = desc.train(adata,
                   dims=[adata.shape[1], 128, 32],
                   tol=0.001,
                   pretrain_epochs=300,
                   louvain_resolution=[0.8],
                   save_dir=out_dir,
                   use_ae_weights=False,
                   do_tsne=True,
                   do_umap=False,
                   use_GPU=True,
                   num_Cores=1,
                   num_Cores_tsne=4,
                   save_encoder_weights=False)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

y_pred = np.asarray(adata.obs['desc_0.8'], dtype=int)
embedding = np.array(adata.obsm['X_Embeded_z0.8'])
k = len(np.unique(y_pred))
nmi, ari = calculate_metric(y, y_pred)

print(ari)
print(nmi)
print(k)



base = os.path.splitext(os.path.basename(args.input_file))[0]
out_file = os.path.join(out_dir, f"results_{base}.npz")
np.savez(out_file, ARI=ari, NMI=nmi, K=k, Embedding=embedding, Clusters=y_pred, Time_use=run_time)

