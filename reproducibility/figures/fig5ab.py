"""Figure 5 (a, b) -- islet-infiltrating CD4 T cells: (a) UMAP of the predicted
clusters at k = 4, 7 and 13 plus the curated annotation, (b) Sankey of how cells
move across those resolutions.

Panels (c)-(f) are in fig5cf.r.

    python fig5ab.py --panel a
    python fig5ab.py --panel b --k-layers 4,7,13
    python fig5ab.py --panel all
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE = Path("/Volumes/SSD/MCW/Research/Aim 1")
RUN_DIR = BASE / "DMVAE/1aelr_0_001_aep_30_lrnn_0_001_beta_0_3"
LABEL_FILE = "CD4_with_Treg_label.txt"
# Beside the run, not relative to the cwd: fig5cf.r reads the cluster_colors_k*.csv
# written here so panel (c) uses the same colours as panel (a).
OUT_DIR = RUN_DIR / "umap_plots"

# k=8 has one empty cluster, so it renders as the 7 clusters the manuscript reports.
K_LAYERS = [4, 8, 13]

UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

POINT_SIZE = 1
FIGSIZE = (8, 8)
LEGEND_FS = 24
LEGEND_TITLE_FS = 26
PANEL_TITLE_FS = 30

SPLIT_SHADES = [-0.55, 0.55, -0.75, 0.75]

NUM_NAMES = {
    0: '0-Memory',
    1: '1-Naive',
    3: '3-Il21+early effect/T fh-like',
    4: '4-Il21+Th1',
    5: '5-Effector memory',
    6: '6-Proliferating',
    7: '7-Acinar contamination',
    8: '8-Naive',
}

DISTINCT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3",
]


def annotation_display(label_str):
    s = str(label_str)
    if s.startswith("treg"):
        return "Treg_" + s.split("_")[1]
    return NUM_NAMES.get(int(s), s)


def load_true_labels(path):
    with open(path, "r") as f:
        lines = [line.strip().strip('"').strip("'") for line in f if line.strip()]
    try:
        labels = np.array([int(x) for x in lines])
        return labels, None, np.array([str(x) for x in labels])
    except ValueError:
        pass
    uniq = sorted(set(lines), key=lambda x: (x.isdigit(), x))
    lab_to_int = {lab: i for i, lab in enumerate(uniq)}
    return np.array([lab_to_int[x] for x in lines]), uniq, np.array(lines)


def _shade(hex_color, factor):
    if not isinstance(hex_color, str):
        rgb = np.array(hex_color[:3], dtype=float)
    else:
        h = hex_color.lstrip("#")[:6]
        rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    if factor >= 0:
        return tuple(rgb + (1.0 - rgb) * factor)
    return tuple(rgb * (1.0 + factor))


def _to_hex(c):
    if isinstance(c, str):
        return c
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


def matched_cluster_colors(assign_k, disp, order, color_of):
    from scipy.optimize import linear_sum_assignment

    clusters = sorted(np.unique(assign_k))
    w = np.zeros((len(clusters), len(order)), dtype=np.int64)
    for ci, c in enumerate(clusters):
        m = assign_k == c
        for oi, lab in enumerate(order):
            w[ci, oi] = int(np.sum(m & (disp == lab)))

    rows, cols = linear_sum_assignment(-w)
    mapping = {int(clusters[r]): int(c) for r, c in zip(rows, cols)}
    for ci, c in enumerate(clusters):
        mapping.setdefault(int(c), int(np.argmax(w[ci])) if w[ci].sum() else -1)

    by_lab = {}
    for c in clusters:
        by_lab.setdefault(mapping[int(c)], []).append((int(c), int(np.sum(assign_k == c))))

    out = {}
    for oi, members in by_lab.items():
        members.sort(key=lambda t: -t[1])
        base = color_of.get(order[oi], "#cccccc") if 0 <= oi < len(order) else "#cccccc"
        for rank, (c, _) in enumerate(members):
            out[c] = _shade(base, 0.0 if rank == 0 else
                            SPLIT_SHADES[(rank - 1) % len(SPLIT_SHADES)])
    return [out[int(c)] for c in clusters], clusters


def _style_square(ax, order, color_of, title="Clusters", panel_title=None):
    if panel_title:
        ax.set_title(panel_title, fontsize=PANEL_TITLE_FS)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=color_of[lab], edgecolor="gray", label=str(lab))
                 for lab in order],
        loc="center left", bbox_to_anchor=(1, 0.5), title=title,
        fontsize=LEGEND_FS, title_fontsize=LEGEND_TITLE_FS,
    )


def _figsize_for(n_entries):
    legend_in = 0.42 * n_entries + 0.8
    return (FIGSIZE[0], max(FIGSIZE[1], legend_in))


def panel_a(out_dir: Path, run_dir: Path, label_file: str | Path,
            k_layers=None) -> None:
    k_layers = k_layers or K_LAYERS
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(run_dir)

    z_mean = np.loadtxt(run_dir / "z_mean.txt")
    if z_mean.ndim == 1:
        z_mean = z_mean.reshape(-1, 1)
    n_samples = z_mean.shape[0]

    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                            metric="euclidean", random_state=UMAP_RANDOM_STATE)
    xy = np.asarray(reducer.fit_transform(z_mean))[:, :2]

    label_path = Path(label_file)
    if not label_path.is_absolute() and not label_path.parent.name:
        label_path = run_dir / label_path
    _, _, y_display = load_true_labels(label_path)
    if len(y_display) != n_samples:
        raise SystemExit(f"True label length {len(y_display)} != z_mean rows {n_samples}")

    with open(run_dir / "assignments_all_k.json") as f:
        assignments_all_k = json.load(f)

    disp = np.array([annotation_display(s) for s in y_display], dtype=object)
    nums = sorted({int(s) for s in y_display if not str(s).startswith("treg")})
    tregs = sorted({int(str(s).split("_")[1]) for s in y_display if str(s).startswith("treg")})
    order = [NUM_NAMES.get(n, str(n)) for n in nums] + [f"Treg_{t}" for t in tregs]
    colors = (DISTINCT_COLORS[:len(order)] if len(order) <= len(DISTINCT_COLORS)
              else [plt.cm.tab20(i / max(1, len(order) - 1)) for i in range(len(order))])
    color_of = {lab: colors[i] for i, lab in enumerate(order)}

    fig_t, ax_t = plt.subplots(1, 1, figsize=_figsize_for(len(order)))
    for lab in order:
        m = disp == lab
        ax_t.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, color=color_of[lab])
    _style_square(ax_t, order, color_of, panel_title="Annotated label")
    fig_t.subplots_adjust(left=0.05, right=0.72, bottom=0.05, top=0.95)
    true_path = out_dir / "umap_true.png"
    fig_t.savefig(true_path, bbox_inches="tight", dpi=300)
    plt.close(fig_t)
    print(f"Saved {true_path}")

    layer_order, _, _ = sankey_layer_order(assignments_all_k, k_layers)

    for k_val in k_layers:
        assign_k = np.array(assignments_all_k[str(k_val)], dtype=int)
        if len(assign_k) != n_samples:
            print(f"Skipping k={k_val}: assignment length {len(assign_k)} != {n_samples}")
            continue

        # Drop empty clusters and renumber contiguously from 0.
        present = np.unique(assign_k)
        if len(present) < k_val:
            print(f"k={k_val}: {k_val - len(present)} empty cluster(s) removed; "
                  f"{len(present)} non-empty cluster(s) remain")
        remap = {int(old): new for new, old in enumerate(present)}
        assign_k = np.array([remap[int(v)] for v in assign_k], dtype=int)

        cols_p, uniq_p = matched_cluster_colors(assign_k, disp, order, color_of)
        fig_p, ax_p = plt.subplots(1, 1, figsize=_figsize_for(len(uniq_p)))
        col_map = {u: cols_p[i] for i, u in enumerate(uniq_p)}
        for u in uniq_p:
            m = assign_k == u
            ax_p.scatter(xy[m, 0], xy[m, 1], s=POINT_SIZE, color=col_map[u])
        _style_square(ax_p, list(uniq_p), col_map,
                      panel_title=f"k={len(uniq_p)}")
        fig_p.subplots_adjust(left=0.05, right=0.72, bottom=0.05, top=0.95)
        pred_path = out_dir / f"umap_predicted_k{k_val}.png"
        fig_p.savefig(pred_path, bbox_inches="tight", dpi=300)
        plt.close(fig_p)
        print(f"Saved {pred_path}")

        # fig5cf.r reads these colours; `rank` is the Sankey position.
        rank_of = {c: r for r, c in enumerate(layer_order.get(str(k_val), []))}
        cmap_path = out_dir / f"cluster_colors_k{k_val}.csv"
        with open(cmap_path, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["cluster", "color", "rank"])
            for u in uniq_p:
                wtr.writerow([int(u), _to_hex(col_map[u]),
                              rank_of.get(int(u), int(u))])
        print(f"Saved {cmap_path}")


SANKEY_W, SANKEY_H = 1200, 1950
SANKEY_FS = 72
SANKEY_SCALE = 3.0


# Layer order as RENDERED: plotly's "snap" nudges node positions, so what it
# draws can differ from the barycentre order below. None = use the computed one.
SANKEY_NODE_ORDER = {
    "4":  [0, 1, 2, 3],
    "8":  [3, 5, 6, 2, 0, 4, 1],
    "13": [4, 10, 12, 0, 2, 11, 3, 7, 6, 5, 8, 9, 1],
}


def sankey_layer_order(assignments, k_layers):
    """{k: [cluster ids top-to-bottom]}, over the renumbered non-empty ids."""
    raw = [np.asarray(assignments[str(k)], dtype=int) for k in k_layers]
    present = [sorted(np.unique(lab).tolist()) for lab in raw]
    layers = [np.array([{r: p for p, r in enumerate(ids)}[int(v)] for v in lab],
                       dtype=int)
              for ids, lab in zip(present, raw)]
    layer_n = [len(ids) for ids in present]
    trans = [Counter(zip(layers[li].tolist(), layers[li + 1].tolist()))
             for li in range(len(layer_n) - 1)]
    order = _barycenter_order(layer_n, trans)

    # An override wins only if it is a permutation of that layer's clusters.
    for i, k in enumerate(k_layers):
        want = SANKEY_NODE_ORDER.get(str(k))
        if want is None:
            continue
        if sorted(want) == sorted(range(layer_n[i])):
            order[i] = list(want)
        else:
            print(f"  [warn] SANKEY_NODE_ORDER['{k}'] is not a permutation of "
                  f"0..{layer_n[i] - 1}; using the computed order")

    return {str(k): order[i] for i, k in enumerate(k_layers)}, layers, layer_n


def _barycenter_order(layer_n, trans_counts, sweeps=4):
    """Order within each layer that minimises crossings (barycentre sweeps)."""
    order = [list(range(n)) for n in layer_n]

    def positions(li):
        return {c: (rank + 0.5) / layer_n[li] for rank, c in enumerate(order[li])}

    def reorder(li, ref_li, forward):
        ref_pos, cur_pos = positions(ref_li), positions(li)
        tc = trans_counts[min(li, ref_li)]
        bary = {}
        for c in range(layer_n[li]):
            num = den = 0.0
            for (ca, cb), n in tc.items():
                here, there = (cb, ca) if forward else (ca, cb)
                if here == c:
                    num += ref_pos[there] * n
                    den += n
            bary[c] = (num / den) if den > 0 else cur_pos[c]
        order[li] = sorted(range(layer_n[li]), key=lambda c: bary[c])

    for _ in range(sweeps):
        for li in range(1, len(layer_n)):
            reorder(li, li - 1, forward=True)
        for li in range(len(layer_n) - 2, -1, -1):
            reorder(li, li + 1, forward=False)
    return order


def panel_b(out_dir: Path, run_dir: Path, label_file, k_layers=None) -> None:
    """Sankey of cells across resolutions; nodes take the panel (a) colours."""
    import plotly.graph_objects as go

    k_layers = [str(k) for k in (k_layers or K_LAYERS)]
    run_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "assignments_all_k.json") as f:
        assignments = json.load(f)
    missing = [k for k in k_layers if k not in assignments]
    if missing:
        raise SystemExit(f"Requested k not found: {', '.join(missing)}")

    n_cells = len(assignments[k_layers[0]])
    for k in k_layers:
        if len(assignments[k]) != n_cells:
            raise SystemExit(f"Length mismatch at k={k}")

    # Renumbered over non-empty clusters, so k=8 is labelled k7_0..k7_6.
    order_by_k, layers, layer_n = sankey_layer_order(assignments, k_layers)
    order = [order_by_k[k] for k in k_layers]
    n_layers = len(k_layers)
    trans = [Counter(zip(layers[li].tolist(), layers[li + 1].tolist()))
             for li in range(n_layers - 1)]

    rank = [{c: r for r, c in enumerate(order[li])} for li in range(n_layers)]
    offset = np.cumsum([0] + layer_n[:-1]).tolist()

    def node_idx(li, c):
        return offset[li] + rank[li][int(c)]

    node_labels = [None] * sum(layer_n)
    node_x = [0.0] * sum(layer_n)
    for li in range(n_layers):
        x = 0.01 if n_layers == 1 else 0.01 + 0.98 * li / (n_layers - 1)
        for c in range(layer_n[li]):
            node_labels[node_idx(li, c)] = f"k{layer_n[li]}_{c}"
            node_x[node_idx(li, c)] = x

    # Panel (a)'s colours.
    label_path = Path(label_file)
    if not label_path.is_absolute() and not label_path.parent.name:
        label_path = run_dir / label_path
    _, _, y_display = load_true_labels(label_path)
    disp = np.array([annotation_display(s) for s in y_display], dtype=object)
    nums = sorted({int(s) for s in y_display if not str(s).startswith("treg")})
    tregs = sorted({int(str(s).split("_")[1]) for s in y_display
                    if str(s).startswith("treg")})
    ann_order = [NUM_NAMES.get(n, str(n)) for n in nums] + [f"Treg_{t}" for t in tregs]
    color_of = {lab: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
                for i, lab in enumerate(ann_order)}

    node_colors = ["#cccccc"] * sum(layer_n)
    for li in range(n_layers):
        cols, uniq = matched_cluster_colors(layers[li], disp, ann_order, color_of)
        for c, col in zip(uniq, cols):
            node_colors[node_idx(li, c)] = _to_hex(col)

    sources, targets, values = [], [], []
    for li in range(n_layers - 1):
        for (ca, cb), n in sorted(trans[li].items()):
            if n > 0:
                sources.append(node_idx(li, ca))
                targets.append(node_idx(li + 1, cb))
                values.append(n)

    def rgba(hex_color, alpha=0.45):
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=38, thickness=22, line=dict(color="black", width=0.5),
                  label=node_labels, color=node_colors, x=node_x),
        link=dict(source=sources, target=targets, value=values,
                  color=[rgba(node_colors[s]) for s in sources]),
    ))
    fig.update_layout(font=dict(size=SANKEY_FS, family="Arial", color="black"),
                      height=SANKEY_H, width=SANKEY_W,
                      # Bottom margin clears the last node's centred label.
                      margin=dict(l=20, r=20, t=40, b=90),
                      paper_bgcolor="white", plot_bgcolor="white")

    # Every transition must account for all cells.
    for li in range(n_layers - 1):
        total = sum(v for s, v in zip(sources, values)
                    if offset[li] <= s < offset[li] + layer_n[li])
        if total != n_cells:
            print(f"  WARNING k{layer_n[li]}->k{layer_n[li+1]}: {total} of {n_cells} cells")

    out_path = out_dir / "sankey_k4_k7_k13.png"
    fig.write_image(str(out_path), width=SANKEY_W, height=SANKEY_H,
                    scale=SANKEY_SCALE)
    print(f"Saved {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Render Figure 5 panels a and b (c-f are in fig5cf.r).")
    p.add_argument("--panel", choices=["a", "b", "all"], default="all")
    p.add_argument("--run-dir", default=str(RUN_DIR),
                   help="directory holding z_mean.txt and assignments_all_k.json")
    p.add_argument("--labels-file", default=str(LABEL_FILE),
                   help="name inside --run-dir, or a path elsewhere")
    p.add_argument("--k-layers", default=",".join(str(k) for k in K_LAYERS),
                   help="k values to draw; k=8 renders as 7 non-empty clusters")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args(argv)

    ks = [int(k) for k in args.k_layers.split(",") if k.strip()]

    if args.panel in ("a", "all"):
        panel_a(Path(args.out_dir), Path(args.run_dir), args.labels_file, ks)
    if args.panel in ("b", "all"):
        panel_b(Path(args.out_dir), Path(args.run_dir), args.labels_file, ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
