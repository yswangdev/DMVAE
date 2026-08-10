"""Prepare scGNN input from the generated simulation data.

scGNN's ``PreprocessingscGNN.py --filetype CSV`` expects a CSV expression matrix
with **genes as rows and cells as columns** (gene names in column 0, cell names in
the header row). Our generators write ``simcounts_{i}.csv`` as raw counts
**cells x genes** (no header). This script transposes each rep into a per-rep folder
``<out_root>/<scenario>/sim{i}/counts.csv`` (genes x cells, named), matching the layout
the existing ``scgnn.slurm`` consumes (``DATA_ROOT/sim{i}/counts.csv``).

Usage (simulations, 20 reps -> sim1/ .. sim20/):
    python scgnn_prep.py --data-dir /.../Data/<scenario> \
                         --out-dir /.../scGNN/Data/<scenario> \
                         --rep-start 1 --rep-end 21

Usage (a real dataset, which has no rep index and should not be called "sim"):
    python scgnn_prep.py --data-dir /.../Data/pbmc_4c \
                         --out-dir /.../scGNN/Data/pbmc_4c \
                         --counts-file pbmc_counts.csv --labels-file pbmc_meta.txt \
                         --rep-name pbmc
"""

import argparse
import os

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="Dir with simcounts_{i}.csv + simmeta_{i}.txt")
parser.add_argument("--out-dir", required=True, help="Per-dataset folders are created here")
parser.add_argument("--rep-start", type=int, default=1)
parser.add_argument("--rep-end", type=int, default=21)
parser.add_argument(
    "--counts-file",
    default=None,
    help="Single-dataset mode (real data): exact counts filename, e.g. pbmc_counts.csv. "
    "Overrides the simcounts_{i}.csv pattern and ignores --rep-start/--rep-end.",
)
parser.add_argument("--labels-file", default=None, help="Single-dataset mode: exact label filename")
parser.add_argument(
    "--rep-name",
    default=None,
    help="Single-dataset mode: output subdir name (default: counts filename without extension). "
    "Use it to keep real datasets out of sim{i}/ folders.",
)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

if args.counts_file:
    if not args.labels_file:
        parser.error("--counts-file requires --labels-file")
    jobs = [(args.counts_file, args.labels_file,
             args.rep_name or os.path.splitext(args.counts_file)[0])]
else:
    jobs = [(f"simcounts_{i}.csv", f"simmeta_{i}.txt", f"sim{i}") for i in range(args.rep_start, args.rep_end)]

for counts_name, meta_name, rep_name in jobs:
    counts_csv = os.path.join(args.data_dir, counts_name)
    meta_txt = os.path.join(args.data_dir, meta_name)
    if not os.path.isfile(counts_csv):
        print(f"[SKIP] {rep_name}: missing {counts_csv}")
        continue

    # raw counts: cells x genes (no header) -> transpose to genes x cells
    counts = np.loadtxt(counts_csv, delimiter=",")
    genes_by_cells = counts.T  # genes x cells
    n_genes, n_cells = genes_by_cells.shape

    df = pd.DataFrame(
        genes_by_cells,
        index=[f"Gene{g}" for g in range(n_genes)],
        columns=[f"Cell{c}" for c in range(n_cells)],
    )

    rep_dir = os.path.join(args.out_dir, rep_name)
    os.makedirs(rep_dir, exist_ok=True)
    df.to_csv(os.path.join(rep_dir, "counts.csv"))

    # carry labels alongside for evaluation
    if os.path.isfile(meta_txt):
        labels = np.loadtxt(meta_txt).astype(int)
        np.savetxt(os.path.join(rep_dir, "label.csv"), labels, fmt="%d", delimiter=",")

    print(f"[OK ] {rep_name}: wrote {rep_dir}/counts.csv  ({n_genes} genes x {n_cells} cells)")
