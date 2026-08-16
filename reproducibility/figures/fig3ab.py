"""Figure 3 (a, b) -- clustering performance on the 13 real scRNA-seq datasets:
(a) ARI heatmap, (b) boxplot of absolute k bias. Values are read from NPZ files.

Panels (c) and (d) are in fig3cd.py. This file is also the single source of
the 13-dataset NPZ loader used by fig6.py and figS1_S2.py.

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

DIRECTORY = os.environ.get("DMVAE_DIRECTORY", ".")
OUTPUT_DIR = os.environ.get(
    "FIGURE_OUTPUT_ROOT", os.path.join(DIRECTORY, "results", "dmvae", "figures")
)
RESULTS_ROOT = os.environ.get(
    "REALWORLD_RESULTS_ROOT",
    os.path.join(DIRECTORY, "results"),
)
BEST_AE_ROOT = os.environ.get(
    "DMVAE_BEST_AE_ROOT",
    os.path.join(RESULTS_ROOT, "best_ae_realworld"),
)

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

RESULTS_DIR = {
    "Bach": "Bach", "Human pancreas": "human_p", "Human PBMC": "PBMC",
    "Klein": "Klein", "Mouse hypothalamus": "mouse_h", "Muraro": "Muraro",
    "Plasschaert": "Plass", "QS Limb Muscle": "QS_LM", "QS Trachea": "QS_trachea",
    "Romanov": "Romanov", "Turtle brain": "turtle_b", "Wang Lung": "Wang_Lung",
    "Young": "Young",
}

BEST_AE_DIR = {
    "Bach": "Bach", "Human pancreas": "human_p", "Human PBMC": "PBMC",
    "Klein": "mouse_ES", "Mouse hypothalamus": "mouse_h", "Muraro": "Muraro",
    "Plasschaert": "Plasschaert",
    "QS Limb Muscle": "Quake_Smart-seq2_Limb_Muscle",
    "QS Trachea": "Quake_Smart-seq2_Trachea", "Romanov": "Romanov",
    "Turtle brain": "turtle_b", "Wang Lung": "Wang_Lung", "Young": "Young",
}

NPZ_NAME = {
    "scVI": "scvi", "scGNN": "scgnn", "ADClust": "adclust",
    "scAce": "scace", "scDAC": "scdac",
}

TRUTH_K = {
    "Bach": 8, "Human pancreas": 14, "Human PBMC": 4, "Klein": 4,
    "Mouse hypothalamus": 46, "Muraro": 9, "Plasschaert": 8, "QS Limb Muscle": 6,
    "QS Trachea": 4, "Romanov": 7, "Turtle brain": 15, "Wang Lung": 2, "Young": 11,
}


def archive_path(dataset: str, method: str) -> str:
    """Return the result archive for one real dataset and method."""
    if method == "DMVAE":
        return os.path.join(BEST_AE_ROOT, BEST_AE_DIR[dataset], "dmvae.npz")
    return os.path.join(
        RESULTS_ROOT, RESULTS_DIR[dataset], f"{NPZ_NAME[method]}.npz"
    )


def _last_scalar(data, key: str) -> float:
    values = np.asarray(data[key], dtype=float).ravel()
    if values.size == 0:
        raise ValueError(f"NPZ field {key!r} is empty")
    return float(values[-1])


def read_archive_scores(dataset: str, method: str) -> dict[str, float | int]:
    """Read final ARI, NMI, and estimated K from one method's NPZ archive."""
    path = archive_path(dataset, method)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing result archive for {dataset}/{method}: {path}. "
            "Set REALWORLD_RESULTS_ROOT or DMVAE_BEST_AE_ROOT if needed."
        )

    with np.load(path, allow_pickle=True) as data:
        scores = {}
        for metric in ("ARI", "NMI"):
            key = metric if metric in data.files else f"Best{metric}"
            if key not in data.files:
                raise KeyError(f"{path} does not contain {metric} or Best{metric}")
            scores[metric] = _last_scalar(data, key)

        if method == "DMVAE" and "AdjustedK" in data.files:
            estimated_k = int(_last_scalar(data, "AdjustedK"))
        elif method != "DMVAE" and "K" in data.files:
            estimated_k = int(_last_scalar(data, "K"))
        elif "Clusters" in data.files:
            clusters = data["Clusters"]
            if method == "scAce":
                clusters = np.asarray(
                    clusters.tolist() if clusters.ndim == 0 else clusters
                )[-1]
                if np.asarray(clusters).ndim == 2:
                    clusters = np.asarray(clusters)[-1]
            estimated_k = int(np.unique(np.asarray(clusters).ravel()).size)
        else:
            raise KeyError(f"{path} does not contain AdjustedK, K, or Clusters")

    scores["K"] = estimated_k
    return scores


def load_metric_table(metric: str) -> pd.DataFrame:
    """Load a 13-dataset ARI or NMI table directly from all NPZ archives."""
    metric = metric.upper()
    if metric not in {"ARI", "NMI"}:
        raise ValueError("metric must be 'ARI' or 'NMI'")
    rows = [
        {"Dataset": dataset, **{
            method: read_archive_scores(dataset, method)[metric]
            for method in METHODS
        }}
        for dataset in DATASETS
    ]
    return pd.DataFrame(rows, columns=["Dataset", *METHODS])


def load_estimated_k_table() -> pd.DataFrame:
    """Load estimated cluster counts directly from all NPZ archives."""
    rows = [
        {"Dataset": dataset, **{
            method: read_archive_scores(dataset, method)["K"]
            for method in METHODS
        }}
        for dataset in DATASETS
    ]
    return pd.DataFrame(rows, columns=["Dataset", *METHODS])


def panel_a(out_dir: str) -> None:
    df = load_metric_table("ARI").set_index("Dataset")

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
    df_est_k = load_estimated_k_table().set_index("Dataset")
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
