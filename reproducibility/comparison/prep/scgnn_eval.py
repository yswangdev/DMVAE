"""Evaluate scGNN output against ground-truth labels -> results_sim_{i}.npz.

scGNN writes, into its per-rep output dir, ``*_embedding.csv`` (cell embedding) and a
cluster file ``*_results.txt``. Both are CSV with a HEADER row and a cell-name index
column (Cell0, Cell1, ...), so they must be read with pandas (index_col=0), not a bare
np.loadtxt. scGNN may also filter cells, so we align to ground-truth labels by the cell
name embedded in the index (CellK -> label[K]).

Output: results_sim_{i}.npz with ARI/NMI/K/Clusters/Embedding/Labels (same schema as the
other methods' *_sim.py).

Usage:
    python scgnn_eval.py --scgnn-out <results>/scgnn/<scenario> \
                         --label-root <scGNN>/Data/<scenario> \
                         --out-dir <results>/scgnn/<scenario> --rep-start 1 --rep-end 21
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

parser = argparse.ArgumentParser()
parser.add_argument("--scgnn-out", required=True, help="Root of scGNN per-rep output dirs (sim{i}/)")
parser.add_argument("--label-root", required=True, help="Root holding sim{i}/label.csv")
parser.add_argument("--out-dir", required=True, help="Where to write results_sim_{i}.npz")
parser.add_argument("--rep-start", type=int, default=1)
parser.add_argument("--rep-end", type=int, default=21)
parser.add_argument(
    "--rep-subdir",
    type=str,
    default="sim{i}",
    help="Template for the per-rep scGNN output subdir under --scgnn-out. Sweeps that write "
    "one dir per resolution can pass e.g. 'sim{i}__res0.2'.",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Suffix for output filenames, e.g. '__res0.4', so parameter sweeps do not overwrite each other.",
)
parser.add_argument(
    "--rep-name",
    default=None,
    help="Single-dataset mode (real data): the one output/label subdir name, e.g. 'pbmc'. "
    "Ignores --rep-start/--rep-end/--rep-subdir and writes results_{rep_name}.npz.",
)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


def _read_named_csv(path):
    """Read a scGNN CSV that has a header row + cell-name index column."""
    df = pd.read_csv(path, index_col=0)
    return df


def _cell_ids(index):
    """Map index labels like 'Cell123' (or plain ints) -> integer cell ids."""
    ids = []
    for nm in index:
        m = re.search(r"(\d+)", str(nm))
        ids.append(int(m.group(1)) if m else None)
    return ids


def _load_embedding(rep_out):
    hits = glob.glob(os.path.join(rep_out, "*_embedding.csv"))
    if not hits:
        return None
    df = _read_named_csv(hits[0])
    return df  # index = cell names, columns = embedding dims


def _load_pred_series(rep_out):
    """Return a pandas Series of predicted cluster labels indexed by cell name, or None."""
    hits = glob.glob(os.path.join(rep_out, "*_results.txt"))
    hits += glob.glob(os.path.join(rep_out, "*results*.csv"))
    for h in hits:
        try:
            df = pd.read_csv(h, index_col=0)
        except Exception:
            try:
                df = pd.read_csv(h, header=None, index_col=0)
            except Exception:
                continue
        if df.shape[1] == 0:
            continue
        col = df.iloc[:, -1]                       # last column = cluster label
        try:
            return col.astype(int)
        except Exception:
            return pd.Series(pd.factorize(col)[0], index=col.index)
    return None


def _louvain_from_embedding(emb_df):
    try:
        import scanpy as sc, anndata as ad
        a = ad.AnnData(emb_df.values)
        sc.pp.neighbors(a, use_rep="X")
        sc.tl.louvain(a)
        return pd.Series(np.array(a.obs["louvain"]).astype(int), index=emb_df.index)
    except Exception:
        return None


def run_one(i, rep_name=None):
    """rep_name set -> single real dataset (dirs and results named after it, no rep index)."""
    name = rep_name if rep_name is not None else args.rep_subdir.format(i=i)
    label_dir = rep_name if rep_name is not None else f"sim{i}"
    out_name = f"results_{rep_name}{args.tag}.npz" if rep_name is not None else f"results_sim_{i}{args.tag}.npz"

    rep_out = os.path.join(args.scgnn_out, name)
    label_csv = os.path.join(args.label_root, label_dir, "label.csv")
    if not os.path.isdir(rep_out) or not os.path.isfile(label_csv):
        print(f"[SKIP] {name}: missing scGNN output dir or label.csv")
        return

    labels_all = np.loadtxt(label_csv, delimiter=",").astype(int)  # line K = label of CellK
    emb_df = _load_embedding(rep_out)

    pred = _load_pred_series(rep_out)
    if pred is None and emb_df is not None:
        pred = _louvain_from_embedding(emb_df)          # fallback only if scGNN clusters absent
    if pred is None:
        print(f"[FAIL] {name}: no predictions (no *_results.txt and Louvain fallback unavailable)")
        return

    # Align predictions <-> true labels by cell id parsed from the index.
    cell_ids = _cell_ids(pred.index)
    if all(c is not None for c in cell_ids) and max(cell_ids) < len(labels_all):
        y = labels_all[cell_ids]
        p = pred.values
    else:
        # positional fallback: assume pred is in original cell order
        p = pred.values
        if len(p) != len(labels_all):
            print(f"[FAIL] {name}: len(pred)={len(p)} != len(labels)={len(labels_all)} and no usable cell ids")
            return
        y = labels_all

    ari = ARI(y, p); nmi = NMI(y, p, average_method="arithmetic"); k = len(np.unique(p))
    emb = emb_df.values if emb_df is not None else np.array([])
    np.savez(
        os.path.join(args.out_dir, out_name),
        ARI=ari, NMI=nmi, K=k, Clusters=p, Embedding=emb, Labels=y,
    )
    print(f"[SAVE] {name}: ARI={ari:.4f} NMI={nmi:.4f} k={k}  (n={len(p)})")


if args.rep_name:
    try:
        run_one(None, rep_name=args.rep_name)
    except Exception as e:
        print(f"[FAIL] {args.rep_name}: {e}")
else:
    for i in range(args.rep_start, args.rep_end):
        try:
            run_one(i)
        except Exception as e:
            print(f"[FAIL] {i}: {e}")
