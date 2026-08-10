"""Figure 2 -- DMVAE on simulated data: (a) ARI across the four scenarios for
six methods, (b) cascading Sankey true -> k=8 -> k=9 -> k=10, (c) UMAP at those
three k plus the ground truth.

    python fig2.py --panel a
    python fig2.py --panel b
    python fig2.py --panel c
    python fig2.py --panel all
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

N_FILES = 20
METHODS = ["scVI", "scGNN", "ADClust", "scACE", "scDAC", "DMVAE"]

# on-disk scenario -> manuscript label. They now agree, so this is the identity;
# it stays a map because the panels index scenarios by the manuscript label.
SCENARIO_MAP = {
    "s01_latent_gmm_k8": "s01",
    "s02_latent_gmm_k6": "s02",
    "s03_batch_effect":  "s03",
    "s04_mixed_hard":    "s04",
}
DISPLAY_ORDER = list(SCENARIO_MAP.values())

SCDAC_BASE = "/scratch/g/chlin/Yushu/scDAC/scDAC/data/"
COMP_ROOT = "/scratch/g/chlin/Yushu/results/Comparison_methods/"
DMVAE_ROOT = "/scratch/g/chlin/Yushu/results/dmvae/"
# All server-rendered figures land in one place.
OUT_ROOT = "/scratch/g/chlin/Yushu/results/dmvae/figures"

COMP_SUBDIR = {
    "scACE":   "scace",
    "scVI":    "scvi",
    "ADClust": "adclust",
    "scGNN":   "scgnn",
}

DMVAE_DIRS = {
    "s03_high_dropout": "s03_high_dropout/pretrain/aeLR_1e_4_aeEp_40_lrNN_5e_5_beta_0p1",
    "s04_many_equal":   "s04_many_equal/pretrain/aeLR_1e_4_aeEp_20_lrNN_5e_4_beta_0p1",
    "s07_small_cells":  "s07_small_cells/pretrain/aeLR_1e_4_aeEp_40_lrNN_1e_3_beta_0p1",
    "s08_mixed_hard":   "s08_mixed_hard/pretrain/aeLR_1e_4_aeEp_20_lrNN_5e_4_beta_0p1",
}

MIXED_SCENARIO = "s08_mixed_hard"
MIXED_RUN_DIR = os.path.join(DMVAE_ROOT, DMVAE_DIRS[MIXED_SCENARIO], "sim1")
TRUE_LABEL_PATH = "/scratch/g/chlin/Yushu/Data/s08_mixed_hard/simmeta_1.txt"
K_STAGES = [8, 9, 10]

PALETTE = ["#8BBDB5", "#508DAB", "#3A528E", "#F39B7F", "#E64B35", "#00A087"]
FLIERPROPS = dict(marker="o", markersize=2, markerfacecolor="black",
                  markeredgecolor="black", alpha=0.8)

LABEL_FS = 20
TICK_FS = 18
LEGEND_FS = 18
PANEL_TITLE_FS = 20

FIGSIZE_BOX = (14, 5)
FIGSIZE_UMAP = (9, 9)
DPI = 600

SANKEY_NODE_FS = 32
SANKEY_HEADER_FS = 38
SANKEY_W, SANKEY_H = 800, 900


def load_metrics(path: str) -> tuple[float, float]:
    try:
        d = np.load(path, allow_pickle=True)
        return (float(np.mean(np.array(d["ARI"]))),
                float(np.mean(np.array(d["NMI"]))))
    except FileNotFoundError:
        return (np.nan, np.nan)


def collect_results(scenario_map: dict[str, str] = SCENARIO_MAP) -> pd.DataFrame:
    records: list[dict] = []
    missing: list[str] = []

    def add(disp: str, method: str, path: str) -> None:
        ari, nmi = load_metrics(path)
        if np.isnan(ari):
            missing.append(path)
        records.append({"Scenario": disp, "Method": method, "ARI": ari, "NMI": nmi})

    for scenario, disp in scenario_map.items():
        for i in range(1, N_FILES + 1):
            add(disp, "scDAC", f"{SCDAC_BASE}{scenario}_sim{i}/scdac.npz")
            for method, sub in COMP_SUBDIR.items():
                add(disp, method,
                    os.path.join(COMP_ROOT, sub, scenario, f"results_sim_{i}.npz"))
            add(disp, "DMVAE",
                os.path.join(DMVAE_ROOT, DMVAE_DIRS[scenario], f"sim{i}", "dmvae.npz"))

    # A missing archive silently becomes a nan and drops out of the boxes, so
    # report it rather than quietly drawing a panel from fewer replicates.
    if missing:
        print(f"WARNING: {len(missing)} of {len(records)} result files not found, "
              f"e.g. {missing[0]}")

    return pd.DataFrame(records)


def make_boxplot(df: pd.DataFrame, metric: str, out_dir: str) -> None:
    """Grouped boxplot: scenarios on x, one box per method within each group.

    Boxes are positioned by hand rather than with seaborn's ``gap=``, so the
    spacing between methods (intra_gap) and between scenarios (group_gap) is set
    independently and the panel renders identically on any seaborn version.
    """
    import matplotlib.patches as mpatches

    n_m = len(METHODS)
    box_w, intra_gap, group_gap = 0.6, 0.25, 1.6
    step = box_w + intra_gap
    group_step = n_m * step + group_gap

    fig, ax = plt.subplots(figsize=FIGSIZE_BOX, dpi=DPI)
    offsets = (np.arange(n_m) - (n_m - 1) / 2) * step
    centers = [gi * group_step for gi in range(len(DISPLAY_ORDER))]

    for gi, disp in enumerate(DISPLAY_ORDER):
        for mi, method in enumerate(METHODS):
            vals = df[(df.Scenario == disp) & (df.Method == method)][metric].dropna().values
            if len(vals) == 0:
                continue
            bp = ax.boxplot(vals, positions=[centers[gi] + offsets[mi]], widths=box_w,
                            patch_artist=True, flierprops=FLIERPROPS,
                            medianprops=dict(color="black", linewidth=1),
                            boxprops=dict(linewidth=0.8),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8))
            for patch in bp["boxes"]:
                patch.set_facecolor(PALETTE[mi])

    ax.set_xticks(centers)
    ax.set_xticklabels(DISPLAY_ORDER, fontsize=TICK_FS)
    ax.set_xlim(centers[0] - group_step / 2, centers[-1] + group_step / 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    handles = [mpatches.Patch(facecolor=PALETTE[i], label=m) for i, m in enumerate(METHODS)]
    ax.legend(handles=handles, title="", loc="lower center", bbox_to_anchor=(0.5, -0.24),
              ncol=n_m, frameon=False, fontsize=LEGEND_FS)
    sns.despine()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{metric}_boxplot.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/{metric}_boxplot.png")


def panel_a(out_dir: str) -> None:
    make_boxplot(collect_results(), "ARI", out_dir)


def hex_to_rgba(h: str, a: float = 0.5) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def _to_hex(c) -> str:
    """Hex string from either a hex string or an RGB float tuple."""
    if isinstance(c, str):
        return c
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


GT_PALETTE = [
    "#D8E6BE", "#8BBDB5", "#508DAB", "#3A528E",
    "#E64B35", "#F39B7F", "#00A087", "#7E6148",
    "#B09C85", "#91D1C2", "#DC0000", "#4DBBD5",
]


def panel_b(out_dir: str, run_dir: str = MIXED_RUN_DIR,
            true_label_path: str = TRUE_LABEL_PATH, write_html: bool = False) -> None:
    import plotly.graph_objects as go

    y_true = np.array([int(l.strip()) for l in open(true_label_path)])
    with open(os.path.join(run_dir, "assignments_all_k.json")) as f:
        assignments = json.load(f)

    stages = [("True", y_true)] + [(f"k={k}", np.array(assignments[str(k)])) for k in K_STAGES]

    node_index, node_labels, node_colors = {}, [], []
    for si, (_, labs) in enumerate(stages):
        # Same colours as panel (c), so a Sankey node and its UMAP cluster match.
        color_of = build_shared_colors(y_true, labs)
        for lab in sorted(np.unique(labs)):
            node_index[(si, lab)] = len(node_labels)
            node_labels.append(str(lab))
            node_colors.append(_to_hex(color_of[lab]))

    source, target, value, link_colors = [], [], [], []
    for si in range(len(stages) - 1):
        a, b = stages[si][1], stages[si + 1][1]
        for la in sorted(np.unique(a)):
            for lb in sorted(np.unique(b)):
                c = int(np.sum((a == la) & (b == lb)))
                if c > 0:
                    sidx = node_index[(si, la)]
                    source.append(sidx)
                    target.append(node_index[(si + 1, lb)])
                    value.append(c)
                    link_colors.append(hex_to_rgba(node_colors[sidx], 0.5))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=12, line=dict(color="black", width=0.5),
                  label=node_labels, color=node_colors),
        link=dict(source=source, target=target, value=value, color=link_colors),
    ))
    fig.update_layout(
        font=dict(size=SANKEY_NODE_FS, family="Arial", color="black"),  # node labels
        height=SANKEY_H, width=SANKEY_W, margin=dict(l=30, r=30, t=110, b=30),
        annotations=[
            dict(x=x, y=1.07, xref="paper", yref="paper", showarrow=False,
                 text=name, font=dict(size=SANKEY_HEADER_FS, family="Arial"))
            for x, (name, _) in zip([0.0, 0.34, 0.67, 1.0], stages)
        ],
    )

    if write_html:
        fig.write_html(f"{out_dir}/sankey_s08_cascade.html")
    try:
        fig.write_image(f"{out_dir}/sankey_s08_cascade.png", scale=3)
        print(f"Saved {out_dir}/sankey_s08_cascade.png")
    except Exception as exc:
        print(f"PNG needs kaleido ({exc}); HTML only.")


def _shade(hex_color: str, factor: float) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")[:6]
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    if factor >= 0:
        return tuple(rgb + (1.0 - rgb) * factor)
    return tuple(rgb * (1.0 + factor))


SPLIT_SHADES = [-0.55, 0.55, -0.75, 0.75]


def build_shared_colors(y_true: np.ndarray, labels: np.ndarray) -> dict:
    gt_labels = sorted(np.unique(y_true))
    base = {lab: GT_PALETTE[i % len(GT_PALETTE)] for i, lab in enumerate(gt_labels)}

    by_dom: dict[int, list[tuple[int, int]]] = {}
    for lab in np.unique(labels):
        cells = y_true[labels == lab]
        dom = int(np.bincount(cells).argmax())
        by_dom.setdefault(dom, []).append((int(lab), int(cells.size)))

    color_of = {}
    for dom, members in by_dom.items():
        members.sort(key=lambda t: -t[1])
        for rank, (lab, _) in enumerate(members):
            if rank == 0:
                color_of[lab] = _shade(base.get(dom, "#cccccc"), 0.0)
            else:
                shade = SPLIT_SHADES[(rank - 1) % len(SPLIT_SHADES)]
                color_of[lab] = _shade(base.get(dom, "#cccccc"), shade)
    return color_of


def plot_umap(labels, coords, ax, title, color_of):
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], c=[color_of[lab]],
                   s=0.5, alpha=0.6, rasterized=True)
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FS, family="Arial")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_c(out_dir: str, run_dir: str = MIXED_RUN_DIR,
            true_label_path: str = TRUE_LABEL_PATH) -> None:
    import warnings
    warnings.filterwarnings("ignore")

    y_true = np.array([int(l.strip()) for l in open(true_label_path)])
    with open(os.path.join(run_dir, "assignments_all_k.json")) as f:
        assignments = json.load(f)

    d = np.load(os.path.join(run_dir, "dmvae.npz"), allow_pickle=True)
    emb = d["Embedding"]
    if isinstance(emb, np.ndarray) and emb.dtype == object:
        emb = emb.item()
    if isinstance(emb, dict):
        emb = emb[list(emb.keys())[-1]]
    emb = np.asarray(emb)
    if emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)

    import umap
    coords = umap.UMAP(n_neighbors=15, min_dist=0.1,
                       random_state=0, metric="euclidean").fit_transform(emb)
    assert len(y_true) == coords.shape[0], "cell count mismatch"

    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_UMAP)
    axs = axs.ravel()

    for idx, k in enumerate(K_STAGES):
        y_pred = np.array(assignments[str(k)], dtype=int)
        n_clusters = len(np.unique(y_pred))
        ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 2)
        print(f"k={k}: K={n_clusters}, ARI={ari}")
        plot_umap(y_pred, coords, axs[idx], f"k={k} ARI={ari}",
                  build_shared_colors(y_true, y_pred))

    plot_umap(y_true, coords, axs[3], "True", build_shared_colors(y_true, y_true))

    plt.tight_layout()
    os.makedirs(f"{out_dir}/Figures", exist_ok=True)
    plt.savefig(f"{out_dir}/Figures/UMAP_s08_k8-10.svg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/Figures/UMAP_s08_k8-10.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/Figures/UMAP_s08_k8-10.*")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Figure 2 panels.")
    p.add_argument("--panel", choices=["a", "b", "c", "all"], default="all")
    p.add_argument("--out-dir", default=OUT_ROOT)
    p.add_argument("--run-dir", default=MIXED_RUN_DIR)
    p.add_argument("--labels-file", default=TRUE_LABEL_PATH)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.panel in ("a", "all"):
        panel_a(args.out_dir)
    if args.panel in ("b", "all"):
        panel_b(args.out_dir, args.run_dir, args.labels_file)
    if args.panel in ("c", "all"):
        panel_c(args.out_dir, args.run_dir, args.labels_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
