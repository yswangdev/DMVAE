"""Figure 3 (a, b) -- clustering performance on the 13 real scRNA-seq datasets:
(a) ARI heatmap, (b) boxplot of absolute k bias. Numbers are inline.

Panels (c) and (d) are in fig3cd.ipynb. This file is also the single source of
the 13-dataset tables, imported by fig6.py and figS1_S3.py.

    python fig3ab.py
    python fig3ab.py --panel a --out-dir /path/out
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.ioff()

OUTPUT_DIR = "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers"

METHODS = ["scVI", "scGNN", "ADClust", "scAce", "scDAC", "DMVAE"]
METHODS_DISPLAY = ["scVI", "scGNN", "ADClust", "scACE", "scDAC", "DMVAE"]
PALETTE = ["#8BBDB5", "#508DAB", "#3A528E", "#F39B7F", "#E64B35", "#00A087"]

TITLE_FS = 24
LABEL_FS = 20
TICK_FS = 18
CBAR_FS = 18
FIGSIZE_HEATMAP = (10, 5)
FIGSIZE_BOX = (7, 5)
DPI = 600

DATASETS = [
    "Bach", "Human pancreas", "Human PBMC", "Klein", "Mouse hypothalamus", "Muraro",
    "Plasschaert", "QS Limb Muscle", "QS Trachea", "Romanov", "Turtle brain",
    "Wang Lung", "Young",
]

ARI_DATA = {
    "Dataset": DATASETS,
    "scVI":    [0.49, 0.70, 0.33, 0.62, 0.57, 0.47, 0.29, 0.48, 0.16, 0.30, 0.50, 0.14, 0.47],
    "scGNN":   [0.67, 0.56, 0.10, 0.66, 0.37, 0.50, 0.38, 0.54, 0.23, 0.26, 0.49, 0.22, 0.23],
    "ADClust": [0.85, 0.79, 0.39, 0.73, 0.78, 0.82, 0.85, 0.97, 0.52, 0.33, 0.62, 0.37, 0.38],
    "scAce":   [0.62, 0.90, 0.49, 0.90, 0.84, 0.90, 0.60, 0.64, 0.34, 0.41, 0.71, 0.19, 0.60],
    "scDAC":   [0.80, 0.84, 0.17, 0.55, 0.74, 0.65, 0.34, 0.55, 0.37, 0.36, 0.38, 0.23, 0.45],
    "DMVAE":   [0.92, 0.94, 0.92, 0.84, 0.87, 0.90, 0.95, 0.91, 0.88, 0.71, 0.81, 0.96, 0.56],
}

TRUTH_K = {
    "Bach": 8, "Human pancreas": 14, "Human PBMC": 4, "Klein": 4,
    "Mouse hypothalamus": 46, "Muraro": 9, "Plasschaert": 8, "QS Limb Muscle": 6,
    "QS Trachea": 4, "Romanov": 7, "Turtle brain": 15, "Wang Lung": 2, "Young": 11,
}

ESTIMATED_K_DATA = {
    "Dataset": DATASETS,
    "scVI":    [20, 11, 9, 10, 22, 13, 14, 10, 20, 20, 22, 13, 28],
    "scGNN":   [8, 5, 5, 7, 9, 7, 7, 6, 12, 7, 8, 7, 12],
    "ADClust": [5, 5, 3, 6, 10, 6, 3, 6, 8, 5, 7, 5, 6],
    "scAce":   [9, 5, 7, 7, 9, 7, 3, 5, 8, 13, 5, 10, 10],
    "scDAC":   [16, 9, 38, 8, 17, 10, 17, 6, 8, 10, 6, 23, 21],
    "DMVAE":   [11, 10, 7, 8, 13, 11, 9, 7, 4, 8, 15, 3, 11],
}


def panel_a(out_dir: str) -> None:
    df = pd.DataFrame(ARI_DATA).set_index("Dataset")

    plt.figure(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    ax = sns.heatmap(
        df.T,
        cmap="YlGnBu",
        vmin=0, vmax=1,
        linewidths=2,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.45},
    )
    plt.xticks(rotation=45, ha="right", fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)

    cbar = ax.collections[0].colorbar
    middle_ticks = [0.25, 0.50, 0.75]
    cbar.set_ticks(middle_ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in middle_ticks])
    cbar.ax.tick_params(labelsize=CBAR_FS)

    plt.title("ARI", fontsize=TITLE_FS)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/heatmap.png")


def panel_b(out_dir: str) -> None:
    df_est_k = pd.DataFrame(ESTIMATED_K_DATA).set_index("Dataset")
    truth_k_series = pd.Series(TRUTH_K, name="K_true")

    df_k_abs_bias = df_est_k.sub(truth_k_series, axis=0).abs().reset_index()
    df_long = df_k_abs_bias.melt(
        id_vars="Dataset",
        value_vars=METHODS,
        var_name="Method",
        value_name="Abs_K_Bias",
    )
    df_long["Method"] = df_long["Method"].replace(dict(zip(METHODS, METHODS_DISPLAY)))

    plt.figure(figsize=FIGSIZE_BOX, dpi=DPI)
    ax = sns.boxplot(
        data=df_long,
        x="Method",
        y="Abs_K_Bias",
        hue="Method",
        order=METHODS_DISPLAY,
        hue_order=METHODS_DISPLAY,
        dodge=False,
        palette=PALETTE,
        width=0.4,
        showfliers=True,
        whis=0,
        flierprops=dict(
            markersize=2,
            markerfacecolor="black",
            markeredgecolor="black",
            alpha=0.8,
        ),
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    plt.xlabel("")
    plt.ylabel("Absolute K bias", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/k_abs_bias_boxplot.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved {out_dir}/k_abs_bias_boxplot.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Figure 3 panels a and b.")
    p.add_argument("--panel", choices=["a", "b", "all"], default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.panel in ("a", "all"):
        panel_a(args.out_dir)
    if args.panel in ("b", "all"):
        panel_b(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
