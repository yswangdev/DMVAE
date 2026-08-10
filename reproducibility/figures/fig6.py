"""Figure 6 -- robustness and cost: (a) ARI and (b) estimated k across
downsampling levels on Plasschaert, (c)(d)(e) ARI stratified by true k, sample
size and platform, (f) runtime. Numbers are inline.

    python fig6.py
    python fig6.py --panel f --out-dir /path/out
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig3ab import ARI_DATA, DATASETS as DATASETS_13, METHODS, TRUTH_K

plt.ioff()

OUTPUT_DIR = "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers"

PCT_ORDER = ["40%", "60%", "80%", "100%"]
PLASSCHAERT_REF_K = 8

FLIER = dict(markersize=2, markerfacecolor="black", markeredgecolor="black", alpha=0.8)

LABEL_FS = 19
TICK_FS = 16
LEGEND_FS = 18
LEGEND_TITLE_FS = 19

STRAT_LABEL_FS = 23
STRAT_TICK_FS = 18
STRAT_LEGEND_FS = 22

# Panel (f) follows the (c)-(e) scale; its method names sit on the Y axis.
RUNTIME_LABEL_FS = STRAT_LABEL_FS
RUNTIME_TICK_FS = STRAT_TICK_FS

FIGSIZE_BAR = (7.5, 4)
FIGSIZE_BOX = (7.5, 4)
DPI = 600

AXES_W_IN = 6.525
AXES_H_IN = 2.78
BOTTOM_IN = 0.54
TOP_GAP_IN = 0.68


def _fixed_axes_margins(fig_w: float, fig_h: float) -> dict:
    side = (fig_w - AXES_W_IN) / 2.0
    return dict(left=side / fig_w, right=1.0 - side / fig_w,
                bottom=BOTTOM_IN / fig_h, top=1.0 - TOP_GAP_IN / fig_h)



DOWNSAMPLE_ARI = {
    "PCT": PCT_ORDER,
    "scVI":    [0.386, 0.315, 0.312, 0.290],
    "scGNN":   [0.200, 0.343, 0.427, 0.380],
    "ADClust": [0.693, 0.753, 0.861, 0.850],
    "scAce":   [0.742, 0.785, 0.604, 0.600],
    "scDAC":   [0.496, 0.411, 0.305, 0.340],
    "DMVAE":   [0.863, 0.953, 0.874, 0.954],
}

DOWNSAMPLE_K = {
    "PCT": PCT_ORDER,
    "scVI":    [11, 13, 12, 14],
    "scGNN":   [4, 5, 6, 7],
    "ADClust": [3, 3, 3, 3],
    "scAce":   [4, 4, 5, 3],
    "scDAC":   [7, 9, 14, 17],
    "DMVAE":   [6, 8, 9, 9],
}

SAMPLE_SIZES = {
    "Bach": 23184, "Human pancreas": 3605, "Human PBMC": 2652, "Klein": 2717,
    "Mouse hypothalamus": 12089, "Muraro": 2122, "Plasschaert": 6977,
    "QS Limb Muscle": 1090, "QS Trachea": 1350, "Romanov": 2881,
    "Turtle brain": 18664, "Wang Lung": 9519, "Young": 5685,
}

PLATFORMS = {
    "Bach": "10x", "Human pancreas": "inDrop", "Human PBMC": "10x", "Klein": "inDrop",
    "Plasschaert": "inDrop", "QS Limb Muscle": "Smart-seq", "QS Trachea": "Smart-seq",
    "Turtle brain": "Smart-seq", "Wang Lung": "10x", "Young": "10x",
}

RUNTIME_DATA = {
    "Dataset": DATASETS_13,
    "scVI":    [923, 220, 141, 157, 576, 100, 318, 60, 70, 271, 893, 464, 236],
    "scGNN":   [9434, 1013, 773, 806, 6016, 769, 2787, 398, 738, 1078, 7693, 2937, 2702],
    "ADClust": [230, 78, 56, 59, 124, 117, 162, 27, 145, 57, 213, 184, 110],
    "scAce":   [3942, 479, 318, 518, 1647, 367, 1768, 185, 232, 573, 3812, 1432, 1359],
    "scDAC":   [11828, 3186, 3625, 2929, 6552, 2974, 5805, 2704, 2716, 2940, 9828, 6359, 3936],
    "DMVAE":   [1396, 272, 159, 148, 574, 166, 379, 102, 118, 189, 1000, 498, 367],
}


def panel_a(out_dir: str) -> None:
    df_long = pd.DataFrame(DOWNSAMPLE_ARI).melt(
        id_vars="PCT", var_name="Method", value_name="ARI")
    df_long["PCT"] = pd.Categorical(df_long["PCT"], categories=PCT_ORDER, ordered=True)

    plt.figure(figsize=FIGSIZE_BAR, dpi=DPI)
    ax = sns.barplot(data=df_long, x="Method", y="ARI", hue="PCT",
                     hue_order=PCT_ORDER, palette="YlGnBu_r", dodge=True,
                     edgecolor="black", linewidth=0.6, saturation=1)
    ax.set_xlabel("")
    ax.set_ylabel("ARI", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(title="Downsampling percentage", loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
              fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ari_methods_colored_by_percent.png", dpi=600)
    plt.close()
    print(f"Saved {out_dir}/ari_methods_colored_by_percent.png")


def panel_b(out_dir: str, k0: int = PLASSCHAERT_REF_K) -> None:
    dfk_long = pd.DataFrame(DOWNSAMPLE_K).melt(
        id_vars="PCT", var_name="Method", value_name="K")
    dfk_long["PCT"] = pd.Categorical(dfk_long["PCT"], categories=PCT_ORDER, ordered=True)
    dfk_long["dK"] = dfk_long["K"] - k0

    methods = dfk_long["Method"].unique().tolist()
    n_hue = len(PCT_ORDER)
    x = np.arange(len(methods)) * 1.5
    bar_w = 0.8 / n_hue
    offsets = (np.arange(n_hue) - (n_hue - 1) / 2) * bar_w
    colors = sns.color_palette("YlGnBu_r", n_colors=n_hue)

    plt.figure(figsize=FIGSIZE_BAR, dpi=DPI)
    ax = plt.gca()
    for i, pct in enumerate(PCT_ORDER):
        sub = dfk_long[dfk_long["PCT"] == pct].set_index("Method").loc[methods]
        ax.bar(x + offsets[i], sub["dK"].values, bottom=k0,
               width=bar_w * 0.95, color=colors[i], edgecolor="black",
               linewidth=0.6, label=pct)

    ax.axhline(k0, linestyle="--", linewidth=1.0, color="black", zorder=3)
    ax.set_yticks([5, 8, 10, 15])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=TICK_FS)
    ax.set_xlabel("")
    ax.set_ylabel("K", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(title="Downsampling percentage", loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
              fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/k_relative_to_8.png", dpi=600,
                facecolor="white", transparent=False)
    plt.close()
    print(f"Saved {out_dir}/k_relative_to_8.png")


def _strat_boxplot(df_long, hue, out_path, ncol, hue_order=None, fig_w=None,
                   legend_fs=None):
    fig_w = fig_w or FIGSIZE_BOX[0]
    legend_fs = legend_fs or STRAT_LEGEND_FS
    plt.figure(figsize=(fig_w, FIGSIZE_BOX[1]), dpi=DPI)
    ax = sns.boxplot(data=df_long, x="Method", y="ARI", hue=hue,
                     hue_order=hue_order, palette="YlGnBu", width=0.4,
                     showfliers=True, whis=0, flierprops=FLIER)
    plt.xlabel("")
    plt.ylabel("ARI", fontsize=STRAT_LABEL_FS)
    ax.tick_params(axis="x", labelsize=STRAT_TICK_FS)
    ax.tick_params(axis="y", labelsize=STRAT_TICK_FS)
    ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=ncol, frameon=False, handletextpad=0.3, columnspacing=0.8,
              fontsize=legend_fs)
    plt.subplots_adjust(**_fixed_axes_margins(fig_w, FIGSIZE_BOX[1]))
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def _k_to_cat(k: int) -> str:
    if k <= 4:
        return "k<=4"
    return "4<k<=10" if k <= 10 else "k>10"


def panel_c(out_dir: str) -> None:
    df = pd.DataFrame(ARI_DATA)
    df["KCat"] = pd.Categorical(
        df["Dataset"].map({d: _k_to_cat(k) for d, k in TRUTH_K.items()}),
        categories=["k<=4", "4<k<=10", "k>10"], ordered=True)
    _strat_boxplot(
        df.melt(id_vars=["Dataset", "KCat"], value_vars=METHODS,
                var_name="Method", value_name="ARI"),
        "KCat", f"{out_dir}/box_by_kcat.png", ncol=3)


def panel_d(out_dir: str) -> None:
    df = pd.DataFrame(ARI_DATA)
    df["n"] = df["Dataset"].map(SAMPLE_SIZES)
    df["SizeCat"] = np.where(df["n"] < 5000, "n < 5000", "n ≥ 5000")
    _strat_boxplot(
        df.melt(id_vars=["Dataset", "SizeCat"], value_vars=METHODS,
                var_name="Method", value_name="ARI"),
        "SizeCat", f"{out_dir}/box_by_n.png", ncol=2)


def panel_e(out_dir: str) -> None:
    df = pd.DataFrame(ARI_DATA)
    df_plat = df[df["Dataset"].isin(PLATFORMS)].copy()
    df_plat["Platform"] = df_plat["Dataset"].map(PLATFORMS)
    _strat_boxplot(
        df_plat.melt(id_vars=["Dataset", "Platform"], value_vars=METHODS,
                     var_name="Method", value_name="ARI"),
        "Platform", f"{out_dir}/box_by_platform.png", ncol=3)


def panel_f(out_dir: str) -> None:
    df_rt_long = pd.DataFrame(RUNTIME_DATA).melt(
        id_vars="Dataset", value_vars=METHODS,
        var_name="Method", value_name="Runtime_sec")
    df_rt_long["Runtime_min"] = df_rt_long["Runtime_sec"] / 60.0

    plt.figure(figsize=FIGSIZE_BOX, dpi=DPI)
    ax = sns.boxplot(data=df_rt_long, y="Method", x="Runtime_min",
                     order=METHODS, palette="YlGnBu", width=0.5,
                     showfliers=True, whis=0, flierprops=FLIER)
    ax.set_xscale("log")
    ax.set_ylabel("")
    ax.set_xlabel("Runtime (minutes, log scale)", fontsize=RUNTIME_LABEL_FS)
    ax.tick_params(axis="x", labelsize=RUNTIME_TICK_FS)
    ax.tick_params(axis="y", labelsize=RUNTIME_TICK_FS)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/runtime_boxplot.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved {out_dir}/runtime_boxplot.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Figure 6 panels.")
    p.add_argument("--panel", choices=["a", "b", "c", "d", "e", "f", "all"], default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    panels = {"a": panel_a, "b": panel_b, "c": panel_c,
              "d": panel_d, "e": panel_e, "f": panel_f}
    for key, fn in panels.items():
        if args.panel in (key, "all"):
            fn(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
