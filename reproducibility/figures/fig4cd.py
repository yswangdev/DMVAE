"""Figure 4 (c, d) -- Mouse hypothalamus: (c) UMAP of the predicted clusters and
of the curated cell types, (d) Sankey from the 46 curated labels to the 13
predicted clusters.

Panels (a) and (b) are in fig4ab.r. Panel (d) is rendered by
fig4d_sankey.py, called as a module.

    python fig4cd.py --panel c
    python fig4cd.py --panel d --methods dmvae
    python fig4cd.py --panel all
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

OUTPUT_DIR = "/Volumes/SSD/MCW/Research/Aim 1/Documents/Paper_draft/papers"
DATA_LABEL = "/Volumes/SSD/MCW/Research/Aim 1/Data/mouse_h/data_label.txt"
DMVAE_NPZ = "/Volumes/SSD/MCW/Research/Aim 1/Results/mouse_h/dmvae.npz"
UMAP_CACHE = "/Volumes/SSD/MCW/Research/Aim 1/Results/umap_mouseh.npz"

# Only these of the 46 cell types are labelled on the panel.
ANNOTATE_TYPES = [
    "MO", "POPC", "Macro", "Micro", "OPC", "Astro", "GABA9", "GABA8",
    "SCO", "Ependy", "Tany", "Endo1", "Endo2", "IMO", "Glu7", "Glu5",
]

TITLE_FS = 21
LEGEND_FS = 18
LEGEND_TITLE_FS = 20
ANNOT_FS = 16
ONDATA_FS = 16


def _strip_axes(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252",
    "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94",
    "#843c39", "#ad494a", "#d6616b", "#e7969c", "#7b4173", "#a55194",
    "#ce6dbd", "#de9ed6",
]
UNMATCHED_COLOR = "#d9d9d9"


def true_colors(categories) -> list:
    return [PALETTE[i % len(PALETTE)] for i in range(len(categories))]


def match_pred_to_true(y_pred, y_true) -> dict:
    from scipy.optimize import linear_sum_assignment

    pred_ids = np.unique(y_pred)
    true_ids = np.unique(y_true)
    w = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for pi, p in enumerate(pred_ids):
        m = y_pred == p
        for ti, t in enumerate(true_ids):
            w[pi, ti] = int(np.sum(m & (y_true == t)))

    rows, cols = linear_sum_assignment(-w)
    mapping = {int(pred_ids[r]): int(c) for r, c in zip(rows, cols)}
    for pi, p in enumerate(pred_ids):
        mapping.setdefault(int(p), int(np.argmax(w[pi])) if w[pi].sum() else -1)
    return mapping


def matched_pred_colors(y_pred, y_true, categories) -> list:
    mapping = match_pred_to_true(y_pred, y_true)

    used: dict[int, int] = {}
    colors = []
    for cat in categories:
        try:
            p = int(cat)
        except (TypeError, ValueError):
            colors.append(UNMATCHED_COLOR)
            continue
        t_idx = mapping.get(p, -1)
        if t_idx < 0:
            colors.append(UNMATCHED_COLOR)
            continue
        base = PALETTE[t_idx % len(PALETTE)]
        seen = used.get(t_idx, 0)
        used[t_idx] = seen + 1
        colors.append(base if seen == 0 else _lighten(base, 0.25 * min(seen, 2)))
    return colors


def _lighten(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [int(round(c + (255 - c) * factor)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*out)

LEGEND_ENTRY_IN = (0.7 + 0.5) * LEGEND_FS / 72.0
LEGEND_W_IN = 2.6
PANEL_PAD_IN = 0.75


def _stack_figsize(n_entries_max: int, data_aspect: float = 1.0) -> tuple:
    legend_h = LEGEND_ENTRY_IN * n_entries_max + 0.55
    axes_h = max(4.0, legend_h)
    axes_w = axes_h * max(data_aspect, 0.2)
    panel_h = axes_h + PANEL_PAD_IN
    return (axes_w + LEGEND_W_IN, 2.0 * panel_h)


def panel_c(out_dir: str, data_label: str = DATA_LABEL,
            dmvae_npz: str = DMVAE_NPZ, umap_cache: str = UMAP_CACHE) -> None:
    import scanpy as sc
    from matplotlib import patheffects
    from matplotlib.patches import Patch

    label_df = pd.read_csv(data_label, header=None, sep=None, engine="python", dtype=str)
    y_true = np.array([str(x).strip('"') for x in label_df.iloc[:, 0].values])

    dmvae = np.load(dmvae_npz)
    y_pred = np.asarray(dmvae["Clusters"], dtype=int).squeeze()
    emb = np.asarray(dmvae["Embedding"])

    n = min(len(y_pred), len(y_true))
    y_pred, y_true, emb = y_pred[:n], y_true[:n], emb[:n]

    # Drop empty clusters and renumber contiguously from 0.
    present = np.unique(y_pred)
    n_total = int(present.max()) + 1 if present.size else 0
    if len(present) < n_total:
        print(f"DMVAE: {n_total - len(present)} empty cluster(s) removed; "
              f"{len(present)} non-empty cluster(s) remain")
    remap = {int(old): new for new, old in enumerate(present)}
    y_pred = np.array([remap[int(v)] for v in y_pred], dtype=int)

    try:
        cached = np.load(umap_cache, allow_pickle=True)["UMAP"]
        umap_coords = cached.item()["DMVAE"][:n]
    except Exception:
        import umap
        umap_coords = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                metric="euclidean", random_state=0).fit_transform(emb)

    adata = sc.AnnData(emb)
    pred_categories = [str(i) for i in range(len(np.unique(y_pred)))]
    adata.obs["pred"] = pd.Categorical(y_pred.astype(str), categories=pred_categories)
    adata.obs["cell_type"] = pd.Categorical(y_true)
    adata.obsm["X_umap"] = umap_coords

    adata.uns["cell_type_colors"] = true_colors(adata.obs["cell_type"].cat.categories)
    adata.uns["pred_colors"] = matched_pred_colors(
        y_pred, y_true, adata.obs["pred"].cat.categories)

    plt.rcParams["font.family"] = "Arial"
    n_legend_max = max(len(adata.obs["pred"].cat.categories),
                       sum(1 for n in ANNOTATE_TYPES
                           if n in list(adata.obs["cell_type"].cat.categories)))
    xr = float(np.ptp(umap_coords[:, 0])) or 1.0
    yr = float(np.ptp(umap_coords[:, 1])) or 1.0
    fig, axes = plt.subplots(2, 1, figsize=_stack_figsize(n_legend_max, xr / yr))

    sc.pl.umap(adata, color="pred", ax=axes[0], show=False,
               legend_loc="on data", legend_fontsize=ONDATA_FS, size=8)
    leg_left = [Patch(facecolor=adata.uns["pred_colors"][i], edgecolor="gray", label=cat)
                for i, cat in enumerate(adata.obs["pred"].cat.categories)]
    axes[0].legend(handles=leg_left, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   title="Clusters", title_fontsize=LEGEND_TITLE_FS,
                   fontsize=LEGEND_FS)
    _strip_axes(axes[0])
    axes[0].set_title("Predicted clusters", fontsize=TITLE_FS)

    sc.pl.umap(adata, color="cell_type", ax=axes[1], show=False, legend_loc=None, size=8)
    cell_type_vals = adata.obs["cell_type"].values
    umap_xy = adata.obsm["X_umap"]
    cat_list = list(adata.obs["cell_type"].cat.categories)

    for name in ANNOTATE_TYPES:
        mask = (cell_type_vals == name)
        if mask.sum() == 0:
            continue
        x, y = np.median(umap_xy[mask], axis=0)
        axes[1].text(x, y, name, fontsize=ANNOT_FS, fontweight="bold", ha="center", va="center",
                     family="Arial",
                     path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])

    leg_handles = [
        Patch(facecolor=adata.uns["cell_type_colors"][cat_list.index(name)],
              edgecolor="gray", label=name)
        for name in ANNOTATE_TYPES if name in cat_list
    ]
    if leg_handles:
        axes[1].legend(handles=leg_handles, loc="center left",
                       bbox_to_anchor=(1.02, 0.5),
                       title="Cell types", title_fontsize=LEGEND_TITLE_FS,
                       fontsize=LEGEND_FS)
    _strip_axes(axes[1])
    axes[1].set_title("Annotated cell types", fontsize=TITLE_FS)

    plt.tight_layout()

    # Size each axes to a square whose side equals its legend height.
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()

    side_in = 0.0
    for ax in axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        leg_h = leg.get_window_extent().transformed(fig.transFigure.inverted()).height
        side_in = max(side_in, leg_h * h_in)

    if side_in > 0:
        x0 = min(ax.get_position().x0 for ax in axes)
        widest = max(ax.get_position().width for ax in axes) * w_in
        new_w_in = w_in + max(0.0, side_in - widest)
        new_h_in = max(h_in, 2.0 * (side_in + PANEL_PAD_IN))
        fig.set_size_inches(new_w_in, new_h_in)

        h_frac = side_in / new_h_in
        w_frac = side_in / new_w_in
        for i, ax in enumerate(axes):
            centre = 0.75 - 0.5 * i
            ax.set_adjustable("datalim")
            ax.set_position([x0, centre - h_frac / 2.0, w_frac, h_frac])

    out_path = f"{out_dir}/rare_cell.png"
    plt.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def panel_d(argv: list[str]) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fig4d_sankey

    saved = sys.argv
    try:
        sys.argv = ["fig4d_sankey.py"] + argv
        fig4d_sankey.main()
    finally:
        sys.argv = saved
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Render Figure 4 panels c and d (a/b are in fig4ab.r).")
    p.add_argument("--panel", choices=["c", "d", "all"], default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--methods", default="dmvae",
                   help="panel d: comma-separated method keys, or 'all'")
    p.add_argument("--data-dir", default=None,
                   help="panel d: directory holding {method}.npz "
                        "(default: fig4d_sankey.py's own default)")
    p.add_argument("--data-label", default=DATA_LABEL)
    p.add_argument("--dmvae-npz", default=DMVAE_NPZ, help="panel c: result archive")
    p.add_argument("--umap-cache", default=UMAP_CACHE)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.panel in ("c", "all"):
        panel_c(args.out_dir, args.data_label, args.dmvae_npz, args.umap_cache)
    if args.panel in ("d", "all"):
        d_argv = ["--methods", args.methods, "--out-dir", args.out_dir,
                  "--data-label", args.data_label]
        if args.data_dir:
            d_argv += ["--data-dir", args.data_dir]
        panel_d(d_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
