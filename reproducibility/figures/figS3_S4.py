"""Supplementary Figures S3 and S4 -- the ten datasets not in Figure 3c/d.

S3 colours each method's UMAP by its own predicted cluster at the cluster number
it inferred; S4 colours the same coordinates by the curated cell-type label.
Rows: Bach, Human pancreas, Human PBMC, Klein, Muraro, QS Limb Muscle,
QS Trachea, Romanov, Wang Lung, Young.

Everything except the dataset list comes from fig3cd.py, so both figures load,
score and draw their panels the same way.

    python figS3_S4.py
    python figS3_S4.py --panel s3 --out-dir /path/out
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig3cd import OUTPUT_DIR, make_grid

# Same keys as fig3cd.DATASETS. Every method here comes from the current run, so
# nothing is frozen and DMVAE is the best_ae_realworld winner throughout.
#
# Human PBMC is the one dataset whose methods disagree on the cell set: the five
# comparison methods ran on 2,593 cells and DMVAE on 2,500, and neither matches
# the 2,652 rows of pbmc_meta_full.txt. resolve_labels sorts this out -- scGNN's
# barcode-matched Labels reproduce the stored ARI for all five comparison
# methods, and data_celltype.txt does so for DMVAE.
DATASETS = [
    dict(name="Bach", results_dir="Bach", best_ae_dir="Bach",
         data_dir="Bach", dmvae_source="best_ae", frozen=(), umap_tag="bach"),
    dict(name="Human pancreas", results_dir="human_p", best_ae_dir="human_p",
         data_dir="human_p", labels=("h5", "Y"),
         dmvae_source="best_ae", frozen=(), umap_tag="human_p"),
    dict(name="Human PBMC", results_dir="PBMC", best_ae_dir="PBMC",
         data_dir="PBMC", labels=("txt", "pbmc_meta_full.txt"),
         extra_labels={"data_celltype.txt": ("txt", "data_celltype.txt")},
         dmvae_source="best_ae", frozen=(), umap_tag="pbmc"),
    dict(name="Klein", results_dir="Klein", best_ae_dir="mouse_ES",
         data_dir="mouse_e", dmvae_source="best_ae", frozen=(),
         umap_tag="Klein"),
    dict(name="Muraro", results_dir="Muraro", best_ae_dir="Muraro",
         data_dir="Muraro", dmvae_source="best_ae", frozen=(),
         umap_tag="Muraro"),
    dict(name="QS Limb Muscle", results_dir="QS_LM",
         best_ae_dir="Quake_Smart-seq2_Limb_Muscle",
         data_dir="Quake_Smart-seq2_Limb_Muscle",
         dmvae_source="best_ae", frozen=(), umap_tag="QS_LM"),
    dict(name="QS Trachea", results_dir="QS_trachea",
         best_ae_dir="Quake_Smart-seq2_Trachea",
         data_dir="Quake_Smart-seq2_Trachea",
         dmvae_source="best_ae", frozen=(), umap_tag="QS_trachea"),
    dict(name="Romanov", results_dir="Romanov", best_ae_dir="Romanov",
         data_dir="Romanov", dmvae_source="best_ae", frozen=(),
         umap_tag="Romanov"),
    dict(name="Wang Lung", results_dir="Wang_Lung", best_ae_dir="Wang_Lung",
         data_dir="Wang_Lung", dmvae_source="best_ae", frozen=(),
         umap_tag="Wang_Lung"),
    dict(name="Young", results_dir="Young", best_ae_dir="Young",
         data_dir="Young", dmvae_source="best_ae", frozen=(),
         umap_tag="Young"),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Render Supplementary Figures S3 and S4.")
    p.add_argument("--panel", choices=["s3", "s4", "all"], default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--recompute-umap", action="store_true",
                   help="ignore the cached coordinates and rebuild them")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.panel in ("s3", "all"):
        make_grid(f"{args.out_dir}/figS3_umap_inferred_k.png", "pred",
                  args.recompute_umap, datasets=DATASETS)
    if args.panel in ("s4", "all"):
        make_grid(f"{args.out_dir}/figS4_umap_true_k.png", "true",
                  args.recompute_umap, datasets=DATASETS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
