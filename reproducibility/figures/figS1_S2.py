"""Supplementary Figures S1 and S2.

    S1  NMI across the four simulation scenarios -- the counterpart of Figure 2a
    S2  NMI on the 13 real datasets, stratified by k, sample size, and platform

The posterior-k and ARI-only plots remain available as unnumbered diagnostics.
The supplementary UMAP grids (S3 and S4; ten datasets) are in figS3_S4.py.

    python figS1_S2.py --part s1
    python figS1_S2.py --part s2
    python figS1_S2.py --part diagnostics --run-dir /path/to/run
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
from fig3ab import (DATASETS as DATASETS_13, METHODS, PALETTE, TRUTH_K,
                    load_estimated_k_table, load_metric_table)
from fig6 import PLATFORMS, SAMPLE_SIZES

plt.ioff()

# Set DMVAE_DIRECTORY to the directory containing Data/ and results/.
DIRECTORY = os.environ.get("DMVAE_DIRECTORY", ".")
OUTPUT_DIR = os.environ.get(
    "FIGURE_OUTPUT_ROOT",
    os.path.join(DIRECTORY, "results", "dmvae", "figures"),
)

FLIER = dict(markersize=2, markerfacecolor="black", markeredgecolor="black", alpha=0.8)

TITLE_FS = 24
LABEL_FS = 20
TICK_FS = 18
CBAR_FS = 18
LEGEND_FS = 15

FIGSIZE_HEATMAP = (10, 7)
FIGSIZE_BOX = (7, 5)
FIGSIZE_STRAT = (7.5, 4)
FIGSIZE_BAR = (6, 4)
DPI = 600


def _strat_boxplot(df_long, hue, out_path, ncol=2, hue_order=None):
    plt.figure(figsize=FIGSIZE_STRAT, dpi=DPI)
    ax = sns.boxplot(data=df_long, x="Method", y=df_long.columns[-1], hue=hue,
                     hue_order=hue_order, palette="YlGnBu", width=0.4,
                     showfliers=True, whis=0, flierprops=FLIER)
    plt.xlabel("")
    plt.ylabel(df_long.columns[-1], fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.2),
              ncol=ncol, frameon=False, handletextpad=0.3, columnspacing=0.8,
              fontsize=LEGEND_FS)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved {out_path}")


def make_nmi_figures(out_dir: str) -> None:
    df = load_metric_table("NMI")

    plt.figure(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    ax = sns.heatmap(df.set_index("Dataset").T, cmap="YlGnBu", vmin=0, vmax=1,
                     linewidths=2, linecolor="white", square=True,
                     cbar_kws={"shrink": 0.45})
    plt.xticks(rotation=45, ha="right", fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.25, 0.50, 0.75])
    cbar.set_ticklabels([f"{t:.2f}" for t in (0.25, 0.50, 0.75)])
    cbar.ax.tick_params(labelsize=CBAR_FS)
    plt.title("NMI", fontsize=TITLE_FS)
    plt.xlabel(""); plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_nmi.png")
    plt.close()
    print(f"Saved {out_dir}/heatmap_nmi.png")

    df_long = df.melt(id_vars="Dataset", value_vars=METHODS,
                      var_name="Method", value_name="NMI")
    plt.figure(figsize=FIGSIZE_BOX, dpi=DPI)
    ax = sns.boxplot(data=df_long, x="Method", y="NMI", palette=PALETTE, width=0.4,
                     showfliers=True, whis=0, flierprops=FLIER)
    plt.xlabel(""); plt.ylabel("NMI", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    plt.title("NMI", fontsize=TITLE_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/box_rw_nmi.png")
    plt.close()
    print(f"Saved {out_dir}/box_rw_nmi.png")

    df["n"] = df["Dataset"].map(SAMPLE_SIZES)
    df["SizeCat"] = np.where(df["n"] < 5000, "n < 5000", "n ≥ 5000")
    df_size_long = df.melt(
        id_vars=["Dataset", "SizeCat"], value_vars=METHODS,
        var_name="Method", value_name="NMI")
    _strat_boxplot(
        df_size_long,
        "SizeCat", f"{out_dir}/box_by_n_nmi.png", ncol=2)

    df_plat = df[df["Dataset"].isin(PLATFORMS)].copy()
    df_plat["Platform"] = df_plat["Dataset"].map(PLATFORMS)
    df_platform_long = df_plat.melt(
        id_vars=["Dataset", "Platform"], value_vars=METHODS,
        var_name="Method", value_name="NMI")
    _strat_boxplot(
        df_platform_long,
        "Platform", f"{out_dir}/box_by_platform_nmi.png", ncol=3)

    def k_to_cat(k):
        if k <= 4:
            return "small (k ≤ 4)"
        return "medium (5 ≤ k ≤ 10)" if k <= 10 else "large (k > 10)"

    df_k = df.copy()
    df_k["KCat"] = pd.Categorical(
        df_k["Dataset"].map({d: k_to_cat(k) for d, k in TRUTH_K.items()}),
        categories=["small (k ≤ 4)", "medium (5 ≤ k ≤ 10)", "large (k > 10)"],
        ordered=True)
    df_k_long = df_k.melt(
        id_vars=["Dataset", "KCat"], value_vars=METHODS,
        var_name="Method", value_name="NMI")
    _strat_boxplot(
        df_k_long,
        "KCat", f"{out_dir}/box_by_kcat_nmi.png", ncol=3)

    # The Supplementary Information presents S2 as one four-panel figure.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    ax_hm = sns.heatmap(
        df.set_index("Dataset")[METHODS].T,
        cmap="YlGnBu", vmin=0, vmax=1, linewidths=1.2, linecolor="white",
        cbar_kws={"shrink": 0.6}, ax=axes[0, 0],
    )
    axes[0, 0].set_title("a", loc="left", fontweight="bold", fontsize=TITLE_FS)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].set_xticklabels(
        axes[0, 0].get_xticklabels(), rotation=52, ha="right",
        rotation_mode="anchor", fontsize=9,
    )
    axes[0, 0].tick_params(axis="y", labelsize=12)
    ax_hm.collections[0].colorbar.ax.tick_params(labelsize=11)

    def combined_box(ax, data, hue, panel, ncol, hue_order=None):
        sns.boxplot(
            data=data, x="Method", y="NMI", hue=hue, hue_order=hue_order,
            palette="YlGnBu", width=0.55, showfliers=True, whis=0,
            flierprops=FLIER, ax=ax,
        )
        ax.set_title(panel, loc="left", fontweight="bold", fontsize=TITLE_FS)
        ax.set_xlabel("")
        ax.set_ylabel("NMI", fontsize=LABEL_FS)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.01),
                  ncol=ncol, frameon=False, fontsize=11)

    combined_box(
        axes[0, 1], df_k_long, "KCat", "b", 3,
        ["small (k ≤ 4)", "medium (5 ≤ k ≤ 10)", "large (k > 10)"],
    )
    combined_box(axes[1, 0], df_size_long, "SizeCat", "c", 2)
    combined_box(axes[1, 1], df_platform_long, "Platform", "d", 3)
    fig.tight_layout()
    combined_path = os.path.join(out_dir, "figS2_nmi_realworld.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined_path}")


def make_sim_nmi_boxplot(out_dir: str) -> None:
    import fig2

    fig2.make_boxplot(
        fig2.collect_results(), "NMI", out_dir,
        filename="figS1_simulation_nmi.png",
    )


DMVAE_ARTIFACT_ROOT = os.environ.get(
    "DMVAE_ARTIFACT_ROOT",
    os.path.join(DIRECTORY, "results", "dmvae", "Simulation"),
)
DEFAULT_PK_RUN = os.path.join(DMVAE_ARTIFACT_ROOT, "s01", "sim1")
DEFAULT_PK_MIN = 6


def make_posterior_k(out_dir: str, run_dir: str = DEFAULT_PK_RUN,
                     k_min: int = DEFAULT_PK_MIN) -> None:
    counts = np.loadtxt(os.path.join(run_dir, "posteriorK_best.txt")).ravel()
    probs = counts / counts.sum()
    K = np.arange(k_min, k_min + len(probs))

    plt.figure(figsize=FIGSIZE_BAR, dpi=300)
    plt.bar(K, probs, color="#508DAB", edgecolor="black", linewidth=0.6)
    plt.xticks(K, fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.xlabel("K", fontsize=LABEL_FS)
    plt.ylabel("S(K)", fontsize=LABEL_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/posteriorK_hist.png", dpi=300)
    plt.close()
    print(f"Saved {out_dir}/posteriorK_hist.png   (k-hat = {K[int(np.argmax(probs))]})")


N_SIMS = 20
GRID_ROWS, GRID_COLS = 4, 5

PK_SCENARIOS = {
    "s01": dict(label="s01", true_k=8),
    "s02": dict(label="s02", true_k=6, a=2, b=15),
    "s03": dict(label="s03", true_k=6),
    "s04": dict(label="s04", true_k=9),
}

GRID_TICK_FS = 9
GRID_SUBTITLE_FS = 11
GRID_AXLABEL_FS = 16
GRID_TITLE_FS = 20
BAR_COLOR = "#508DAB"
BAR_COLOR_TRUE = "#E64B35"


def _load_pk(path: str) -> np.ndarray | None:
    try:
        counts = np.loadtxt(path).ravel()
    except (FileNotFoundError, OSError):
        return None
    if counts.size == 0 or counts.sum() <= 0:
        return None
    return counts / counts.sum()


def make_posterior_k_grid(out_dir: str, scenarios: dict | None = None,
                          dmvae_root: str | None = None) -> None:
    scenarios = scenarios or PK_SCENARIOS
    dmvae_root = dmvae_root or DMVAE_ARTIFACT_ROOT

    for scenario, cfg in scenarios.items():
        fig, axs = plt.subplots(GRID_ROWS, GRID_COLS,
                                figsize=(3.0 * GRID_COLS, 2.4 * GRID_ROWS),
                                dpi=300, sharex=True, sharey=True)
        axs = axs.ravel()

        n_hit = n_seen = 0
        for i in range(1, N_SIMS + 1):
            ax = axs[i - 1]
            probs = _load_pk(os.path.join(dmvae_root, scenario, f"sim{i}",
                                          "posteriorK_best.txt"))
            if probs is None:
                ax.set_title(f"sim{i}", fontsize=GRID_SUBTITLE_FS)
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=GRID_SUBTITLE_FS,
                        color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            k_min = cfg.get("a", cfg["true_k"] - 2)
            k_max = cfg.get("b", cfg["true_k"] + 2)
            K = np.arange(k_min, k_max + 1)
            if len(probs) != len(K):
                raise ValueError(
                    f"{scenario}/sim{i}: expected {len(K)} posterior values "
                    f"for K={K[0]}..{K[-1]}, found {len(probs)}"
                )
            colors = [BAR_COLOR_TRUE if k == cfg["true_k"] else BAR_COLOR for k in K]
            ax.bar(K, probs, color=colors, edgecolor="black", linewidth=0.4)

            k_hat = int(K[int(np.argmax(probs))])
            n_seen += 1
            n_hit += int(k_hat == cfg["true_k"])

            ax.set_title(f"sim{i}  $\\hat{{k}}$={k_hat}", fontsize=GRID_SUBTITLE_FS)
            ax.set_xticks(K)
            ax.tick_params(labelsize=GRID_TICK_FS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(N_SIMS, len(axs)):
            axs[j].axis("off")

        fig.supxlabel("K", fontsize=GRID_AXLABEL_FS)
        fig.supylabel("S(K)", fontsize=GRID_AXLABEL_FS)
        fig.suptitle(
            f"{cfg['label']}  (true K = {cfg['true_k']});  "
            f"$\\hat{{k}}$ = true K in {n_hit}/{n_seen} replicates",
            fontsize=GRID_TITLE_FS,
        )
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

        out_path = f"{out_dir}/S_posteriorK_grid_{cfg['label']}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}   ({n_hit}/{n_seen} replicates peak at K={cfg['true_k']})")


def make_plot_extras(out_dir: str) -> None:
    df_long = load_metric_table("ARI").melt(
        id_vars="Dataset", value_vars=METHODS, var_name="Method", value_name="ARI")
    plt.figure(figsize=FIGSIZE_BOX, dpi=DPI)
    ax = sns.boxplot(data=df_long, x="Method", y="ARI", palette=PALETTE, width=0.4,
                     showfliers=True, whis=0, flierprops=FLIER)
    plt.xlabel(""); plt.ylabel("ARI", fontsize=LABEL_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    plt.title("ARI", fontsize=TITLE_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/box_rw.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved {out_dir}/box_rw.png")

    df_est_k = load_estimated_k_table().set_index("Dataset")
    df_k_bias = df_est_k.sub(pd.Series(TRUTH_K), axis=0).abs()
    vmax_clip = np.percentile(df_k_bias.to_numpy(), 90)

    plt.figure(figsize=FIGSIZE_HEATMAP, dpi=DPI)
    ax = sns.heatmap(df_k_bias.T, cmap="YlGnBu_r", vmin=0, vmax=vmax_clip,
                     linewidths=2, linecolor="white", square=True,
                     cbar_kws={"shrink": 0.45})
    plt.xticks(rotation=45, ha="right", fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    ax.collections[0].colorbar.ax.tick_params(labelsize=CBAR_FS)
    plt.xlabel(""); plt.ylabel("")
    plt.title("Absolute bias of estimated K", fontsize=TITLE_FS)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/k_bias_heatmap.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/k_bias_heatmap.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Supplementary Figures S1 and S2.")
    p.add_argument("--part",
                   choices=["s1", "s2", "diagnostics", "nmi", "simnmi",
                            "pk", "pkgrid", "extras", "all"],
                   default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--run-dir", default=DEFAULT_PK_RUN)
    p.add_argument("--k-min", type=int, default=DEFAULT_PK_MIN)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.part in ("s1", "simnmi", "all"):
        make_sim_nmi_boxplot(args.out_dir)
    if args.part in ("s2", "nmi", "all"):
        make_nmi_figures(args.out_dir)
    if args.part in ("pk", "diagnostics"):
        make_posterior_k(args.out_dir, args.run_dir, args.k_min)
    if args.part in ("pkgrid", "diagnostics"):
        make_posterior_k_grid(args.out_dir)
    if args.part in ("extras", "diagnostics"):
        make_plot_extras(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
