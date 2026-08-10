"""Supplementary Figures S1, S2 and S3.

    S1  NMI across the four simulation scenarios -- the counterpart of Figure 2a
    S2  posterior of k on the simulated data
    S3  NMI on the 13 real datasets, and the ARI stratifications

The supplementary UMAP grids (S4, S5; ten datasets) are in figS4_S5.ipynb.

    python figS1_S3.py --part nmi
    python figS1_S3.py --part simnmi       # NMI counterpart of Figure 2a
    python figS1_S3.py --part pk --run-dir /path/to/run
    python figS1_S3.py --part pkgrid       # 4 panels x 20 replicate histograms
    python figS1_S3.py --part extras
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
from fig3ab import (ARI_DATA, DATASETS as DATASETS_13, ESTIMATED_K_DATA, METHODS,
                  PALETTE, TRUTH_K)
from fig6 import PLATFORMS, SAMPLE_SIZES

plt.ioff()

OUTPUT_DIR = "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers"
SERVER_FIGURES_DIR = "/scratch/g/chlin/Yushu/results/dmvae/figures"

NMI_DATA = {
    "Dataset": DATASETS_13,
    "scVI":    [0.74, 0.84, 0.58, 0.76, 0.75, 0.74, 0.55, 0.75, 0.50, 0.56, 0.73, 0.37, 0.72],
    "scGNN":   [0.77, 0.59, 0.32, 0.72, 0.49, 0.62, 0.56, 0.72, 0.57, 0.33, 0.48, 0.42, 0.37],
    "ADClust": [0.80, 0.80, 0.58, 0.68, 0.78, 0.81, 0.78, 0.95, 0.59, 0.45, 0.73, 0.52, 0.57],
    "scAce":   [0.76, 0.87, 0.60, 0.90, 0.79, 0.87, 0.62, 0.78, 0.60, 0.57, 0.72, 0.38, 0.74],
    "scDAC":   [0.80, 0.84, 0.38, 0.70, 0.78, 0.79, 0.57, 0.74, 0.65, 0.53, 0.52, 0.40, 0.63],
    "DMVAE":   [0.90, 0.88, 0.87, 0.82, 0.76, 0.84, 0.90, 0.85, 0.79, 0.60, 0.76, 0.90, 0.66],
}

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
    df = pd.DataFrame(NMI_DATA)

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
    _strat_boxplot(
        df.melt(id_vars=["Dataset", "SizeCat"], value_vars=METHODS,
                var_name="Method", value_name="NMI"),
        "SizeCat", f"{out_dir}/box_by_n_nmi.png", ncol=2)

    df_plat = df[df["Dataset"].isin(PLATFORMS)].copy()
    df_plat["Platform"] = df_plat["Dataset"].map(PLATFORMS)
    _strat_boxplot(
        df_plat.melt(id_vars=["Dataset", "Platform"], value_vars=METHODS,
                     var_name="Method", value_name="NMI"),
        "Platform", f"{out_dir}/box_by_platform_nmi.png", ncol=3)

    def k_to_cat(k):
        if k <= 4:
            return "small (k<=4)"
        return "medium (5<=k<=10)" if k <= 10 else "large (k>10)"

    df_k = df.copy()
    df_k["KCat"] = pd.Categorical(
        df_k["Dataset"].map({d: k_to_cat(k) for d, k in TRUTH_K.items()}),
        categories=["small (k<=4)", "medium (5<=k<=10)", "large (k>10)"], ordered=True)
    _strat_boxplot(
        df_k.melt(id_vars=["Dataset", "KCat"], value_vars=METHODS,
                  var_name="Method", value_name="NMI"),
        "KCat", f"{out_dir}/box_by_kcat_nmi.png", ncol=3)


def make_sim_nmi_boxplot(out_dir: str) -> None:
    import fig2

    fig2.make_boxplot(fig2.collect_results(), "NMI", out_dir)


DEFAULT_PK_RUN = ("/scratch/g/chlin/Yushu/results/dmvae/s03_high_dropout/pretrain/"
                  "aeLR_1e_4_aeEp_40_lrNN_5e_5_beta_0p1/sim1")


def make_posterior_k(out_dir: str, run_dir: str = DEFAULT_PK_RUN,
                     k_min: int = 5) -> None:
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

DMVAE_DIRS = {
    "s03_high_dropout": "s03_high_dropout/pretrain/aeLR_1e_4_aeEp_40_lrNN_5e_5_beta_0p1",
    "s04_many_equal":   "s04_many_equal/pretrain/aeLR_1e_4_aeEp_20_lrNN_5e_4_beta_0p1",
    "s07_small_cells":  "s07_small_cells/pretrain/aeLR_1e_4_aeEp_40_lrNN_1e_3_beta_0p1",
    "s08_mixed_hard":   "s08_mixed_hard/pretrain/aeLR_1e_4_aeEp_20_lrNN_5e_4_beta_0p1",
}

PK_SCENARIOS = {
    "s03_high_dropout": dict(label="s01", true_k=7),
    "s04_many_equal":   dict(label="s02", true_k=8),
    "s07_small_cells":  dict(label="s03", true_k=5),
    "s08_mixed_hard":   dict(label="s04", true_k=9),
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
                          dmvae_root: str | None = None,
                          dmvae_dirs: dict | None = None) -> None:
    import fig2

    scenarios = scenarios or PK_SCENARIOS
    dmvae_root = dmvae_root or fig2.DMVAE_ROOT
    dmvae_dirs = dmvae_dirs or DMVAE_DIRS

    for scenario, cfg in scenarios.items():
        rel = dmvae_dirs.get(scenario)
        if rel is None:
            print(f"[skip] {scenario}: no DMVAE run directory configured")
            continue

        fig, axs = plt.subplots(GRID_ROWS, GRID_COLS,
                                figsize=(3.0 * GRID_COLS, 2.4 * GRID_ROWS),
                                dpi=300, sharex=True, sharey=True)
        axs = axs.ravel()

        n_hit = n_seen = 0
        for i in range(1, N_SIMS + 1):
            ax = axs[i - 1]
            probs = _load_pk(os.path.join(dmvae_root, rel, f"sim{i}",
                                          "posteriorK_best.txt"))
            if probs is None:
                ax.set_title(f"sim{i}", fontsize=GRID_SUBTITLE_FS)
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=GRID_SUBTITLE_FS,
                        color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            K = np.arange(cfg["true_k"] - 2, cfg["true_k"] + 3)
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
    df_long = pd.DataFrame(ARI_DATA).melt(
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

    df_est_k = pd.DataFrame(ESTIMATED_K_DATA).set_index("Dataset")
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
    p = argparse.ArgumentParser(description="Render supplementary figures.")
    p.add_argument("--part",
                   choices=["nmi", "simnmi", "pk", "pkgrid", "extras", "all"],
                   default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--run-dir", default=DEFAULT_PK_RUN)
    p.add_argument("--k-min", type=int, default=5)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.part in ("nmi", "all"):
        make_nmi_figures(args.out_dir)
    if args.part == "simnmi":
        make_sim_nmi_boxplot(args.out_dir)
    if args.part in ("pk", "all"):
        make_posterior_k(args.out_dir, args.run_dir, args.k_min)
    if args.part in ("pkgrid", "all"):
        make_posterior_k_grid(args.out_dir)
    if args.part in ("extras", "all"):
        make_plot_extras(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
