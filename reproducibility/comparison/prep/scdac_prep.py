"""Prepare scDAC input following the scDAC github (labomics/scDAC) data layout.

Reproduces the on-disk structure of an scDAC dataset (verified against
scDAC/data/chen_10), produced by preprocess.ipynb + preprocess_split.ipynb:

  data/<task>/
  |- feat/
  |   |- feat_dims.csv          # ",rna\\n1,<n_feat>"
  |   `- feat_names_rna.csv      # ",x\\n1,<name>\\n2,<name>..."   (n_feat rows)
  `- subset_0/
      |- cell_name.csv          # ",x\\n1,<cell>\\n2,<cell>..."   (n_cells rows)
      |- mask/
      |   `- rna.csv            # '"","V1",...,"V<n_feat>"' then ONE row: "1",1,1,...,1
      `- vec/
          `- rna/
              |- 0000.csv       # one cell per file: a single row of raw counts (no header)
              `- ...            # zero-padded width = floor(log10(n_cells))+1  (utils.get_name_fmt)
  label.csv                     # one true label per cell (for run.py --label-path)

The per-cell vectors are RAW COUNTS of the (<=)4000 highly-variable features
(scDAC normalizes internally). With our 2000-gene sims all genes are kept.
The task is also registered in scDAC/configs/data.toml.

Usage (simulations, 20 reps -> sim1/ .. sim20/, tasks <scenario>_sim{i}):
    python scdac_prep.py --data-dir /.../Data/<scenario> \
        --out-dir  $SCDAC_DIR/data/<scenario> \
        --rel-prefix data/<scenario> \
        --data-toml $SCDAC_DIR/configs/data.toml \
        --task-prefix <scenario>_sim --n-hvg 4000 --N 512 --rep-start 1 --rep-end 21

Usage (a real dataset, which has no rep index and should not be called "sim"):
    python scdac_prep.py --data-dir /.../Data/pbmc_4c \
        --out-dir  $SCDAC_DIR/data/pbmc_4c \
        --rel-prefix data/pbmc_4c \
        --data-toml $SCDAC_DIR/configs/data.toml \
        --counts-file pbmc_counts.csv --labels-file pbmc_meta.txt \
        --rep-name pbmc --task-name pbmc_4c --n-hvg 4000 --N 512
"""

import argparse
import csv
import fcntl
import math
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True, help="Dir with simcounts_{i}.csv + simmeta_{i}.txt")
parser.add_argument("--out-dir", required=True, help="Absolute scDAC data dir for this scenario ($SCDAC_DIR/data/<scenario>)")
parser.add_argument("--rel-prefix", required=True, help="raw_data_dirs prefix relative to SCDAC root, e.g. data/<scenario>")
parser.add_argument("--data-toml", required=True, help="Path to scDAC configs/data.toml (entries appended)")
parser.add_argument("--task-prefix", default=None, help="Task name prefix, e.g. <scenario>_sim -> <prefix>{i}")
parser.add_argument("--n-hvg", type=int, default=4000)
parser.add_argument("--N", type=int, default=512)
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
    help="Single-dataset mode: data subdir name under --out-dir (default: counts filename "
    "without extension). Use it to keep real datasets out of sim{i}/ folders.",
)
parser.add_argument(
    "--task-name",
    default=None,
    help="Single-dataset mode: scDAC task name registered in data.toml (default: --rep-name)",
)
args = parser.parse_args()

if args.counts_file:
    if not args.labels_file:
        parser.error("--counts-file requires --labels-file")
elif not args.task_prefix:
    parser.error("--task-prefix is required unless --counts-file is given")


def select_hvg(counts):
    """counts: cells x genes raw counts -> column indices of <=n_hvg HVGs."""
    n_genes = counts.shape[1]
    if n_genes <= args.n_hvg:
        return np.arange(n_genes)
    import scanpy as sc
    ad = sc.AnnData(counts.astype(np.float32))
    sc.pp.highly_variable_genes(ad, flavor="seurat_v3", n_top_genes=args.n_hvg)
    return np.where(ad.var["highly_variable"].values)[0]


def write_indexed_col(path, header, values):
    """R write.csv style: header ',<header>' then rows '<i+1>,<value>'."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", header])
        for j, v in enumerate(values):
            w.writerow([j + 1, v])


def toml_has_task(toml_path, task):
    if not os.path.isfile(toml_path):
        return False
    with open(toml_path) as f:
        return any(line.strip() == f"[{task}]" for line in f)


def append_task(toml_path, task, rel_dir):
    block = (
        f"\n[{task}]\n"
        f'raw_data_dirs = [\n    "{rel_dir}",\n]\n'
        "combs = [\n    [ [\"rna\"] ],\n]\n"
        "comb_ratios = [ [ 1 ] ]\n"
        "s_joint =     [ [ 0 ]]\n"
        "train_ratio = 1\n"
        f"N = {args.N}\n"
    )
    with open(toml_path, "a") as f:
        f.write(block)


def register_task(toml_path, task, rel_dir):
    """flock-protected check+append so concurrent scenario jobs can't corrupt data.toml."""
    lock_path = toml_path + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not toml_has_task(toml_path, task):
                append_task(toml_path, task, rel_dir)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def prep_one(counts_name, meta_name, rep_name, task):
    counts_csv = os.path.join(args.data_dir, counts_name)
    meta_txt = os.path.join(args.data_dir, meta_name)
    if not os.path.isfile(counts_csv):
        print(f"[SKIP] {task}: missing {counts_csv}")
        return

    counts = np.loadtxt(counts_csv, delimiter=",")          # cells x genes raw
    if counts.ndim == 1:
        counts = counts[None, :]
    y = np.loadtxt(meta_txt).astype(int)

    hvg = select_hvg(counts)
    mat = counts[:, hvg].astype(int)                        # cells x HVG, raw counts
    n_cells, n_feat = mat.shape

    task_dir = os.path.join(args.out_dir, rep_name)
    feat_dir = os.path.join(task_dir, "feat")
    sub_dir = os.path.join(task_dir, "subset_0")
    mask_dir = os.path.join(sub_dir, "mask")
    vec_dir = os.path.join(sub_dir, "vec", "rna")
    for d in (feat_dir, mask_dir, vec_dir):
        os.makedirs(d, exist_ok=True)

    feat_names = [f"Gene{int(g)}" for g in hvg]
    cell_names = [f"Cell{c}" for c in range(n_cells)]

    # feat/feat_dims.csv : ",rna" then "1,<n_feat>"
    with open(os.path.join(feat_dir, "feat_dims.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["", "rna"]); w.writerow([1, n_feat])
    # feat/feat_names_rna.csv : ",x" then "j,<name>"
    write_indexed_col(os.path.join(feat_dir, "feat_names_rna.csv"), "x", feat_names)
    # subset_0/cell_name.csv : ",x" then "j,<cell>"
    write_indexed_col(os.path.join(sub_dir, "cell_name.csv"), "x", cell_names)

    # subset_0/mask/rna.csv : header "","V1".."V<n_feat>" then ONE all-ones row "1",1,...,1
    with open(os.path.join(mask_dir, "rna.csv"), "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow([""] + [f"V{j+1}" for j in range(n_feat)])
        w.writerow(["1"] + [1] * n_feat)

    # subset_0/vec/rna/<padded>.csv : one cell per file, single row of raw counts (no header)
    width = int(math.floor(math.log10(max(n_cells, 1)))) + 1
    fmt = "%0" + str(width) + "d"
    for c in range(n_cells):
        with open(os.path.join(vec_dir, (fmt % c) + ".csv"), "w", newline="") as f:
            csv.writer(f).writerow(mat[c].tolist())

    # labels: two columns, no header, col1 = integer label -- the pipeline does
    # `[int(row[1]) for row in label_true]`.
    with open(os.path.join(task_dir, "label.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for j, lab in enumerate(y):
            w.writerow([f"Cell{j}", int(lab)])

    # register task in data.toml (idempotent + flock-safe for concurrent scenario jobs)
    rel_dir = f"{args.rel_prefix}/{rep_name}"
    register_task(args.data_toml, task, rel_dir)

    print(f"[OK ] {task}: {n_cells} cells x {n_feat} feat -> {task_dir}  (toml: {rel_dir})")


if args.counts_file:
    rep = args.rep_name or os.path.splitext(args.counts_file)[0]
    jobs = [(args.counts_file, args.labels_file, rep, args.task_name or rep)]
else:
    jobs = [
        (f"simcounts_{i}.csv", f"simmeta_{i}.txt", f"sim{i}", f"{args.task_prefix}{i}")
        for i in range(args.rep_start, args.rep_end)
    ]

for counts_name, meta_name, rep_name, task in jobs:
    try:
        prep_one(counts_name, meta_name, rep_name, task)
    except Exception as e:
        print(f"[FAIL] {task}: {e}")
