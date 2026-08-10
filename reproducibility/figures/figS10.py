"""Supplementary Figure S10 -- p(c = cluster_i | k) panels: the soft cluster
posterior on the DMVAE UMAP.

One panel per predicted cluster i at a chosen k, each cell coloured by its
posterior probability of belonging to that cluster; k panels lay out on the
smallest grid that fits.

Requires a run with --legacy-artifacts, which saves p_c_z_best.npy.

    python figS10.py --results-dir /path/to/mouse_es
    python figS10.py --results-dir /path/to/mouse_es --k 4 \
        --label-names d0,d2,d4,d7
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.ioff()

TITLE_FS = 15
LABEL_FS = 12
TICK_FS = 10
DPI = 600

# Sequential ramp, monotonic in lightness. "grey_red" keeps p~0 recessive grey.
CMAP = "grey_red"

CUSTOM_CMAPS = {
    "grey_red": ["#E6E6E6", "#EFB8A6", "#DD7B62", "#C1392F", "#8E1015"],
}


def resolve_cmap(name):
    if name in CUSTOM_CMAPS:
        return LinearSegmentedColormap.from_list(name, CUSTOM_CMAPS[name])
    return name


def load_run(results_dir, labels_path_override=None):
    """Load the posterior and whatever is available to plot it against."""
    pcz_path = os.path.join(results_dir, "p_c_z_best.npy")
    if not os.path.exists(pcz_path):
        raise SystemExit(
            f"No p_c_z_best.npy in {results_dir}.\n"
            "This figure needs the soft posterior, which older runs did not save "
            "(every saved assignment is an argmax of it, and an argmax cannot be "
            "inverted). Re-run model/run.py --legacy-artifacts to produce it."
        )
    p_c_z = np.load(pcz_path)

    meta_path = os.path.join(results_dir, "p_c_z_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # Fall back to inferring the k-axis offset from the array shape alone.
        meta = {"a": None, "b": p_c_z.shape[2], "shape": list(p_c_z.shape)}

    # Without a UMAP the posterior is shown against the true groups instead.
    umap_path = os.path.join(results_dir, "dmvae_umap_2d.npy")
    xy = np.load(umap_path) if os.path.exists(umap_path) else None

    labels_path = labels_path_override or os.path.join(results_dir, "labels_true.txt")
    y = np.loadtxt(labels_path).astype(int) if os.path.exists(labels_path) else None

    n = p_c_z.shape[0]
    if xy is not None and xy.shape[0] != n:
        raise SystemExit(
            f"Cell count mismatch: UMAP has {xy.shape[0]}, posterior has {n}. "
            "These are not from the same run."
        )
    if y is not None and y.shape[0] != n:
        raise SystemExit(
            f"Cell count mismatch: labels have {y.shape[0]}, posterior has {n}. "
            f"({labels_path} is not the label file this run was trained on.)"
        )
    return p_c_z, meta, xy, y


def check_floor(p_c_z, k_idx, k, b, meta):
    """Report whether the posterior for this k-slice is numerically degenerate.

    GetGamma floors the unnormalised density at 1e-10 before normalising. When
    every component underflows that floor the posterior becomes a flat 1/b, which
    on these panels looks like a uniformly "uncertain" field -- visually identical
    to a genuine graded posterior but carrying no information. The signature is
    the padded components holding (b - k)/b of the mass.
    """
    if k >= b:
        return None
    pad = p_c_z[:, k_idx, k:].sum(axis=1)
    degenerate = (b - k) / b
    frac = float(np.mean(np.abs(pad - degenerate) < 1e-3))
    print(f"  padding mass: mean {pad.mean():.4f} (degenerate would be {degenerate:.4f})")
    print(f"  cells at the degenerate value: {frac:.1%}")
    if frac > 0.01:
        print(
            "  WARNING: some cells have a numerically floored, uniform posterior. "
            "Their apparent uncertainty is an artifact of the 1e-10 floor in "
            "GetGamma, not evidence of a lineage continuum. Exclude them or treat "
            "the affected panels as uninterpretable."
        )
    return frac


def summarise_by_truth(p_c_z, k_idx, k, y, label_names):
    """Mean p(c=i|k) within each true group -- the numeric form of the figure."""
    if y is None:
        return
    groups = np.unique(y)
    names = {g: (label_names[j] if label_names and j < len(label_names) else str(g))
             for j, g in enumerate(groups)}
    print("\n  mean p(c=i | k) by true group (rows: true group, cols: cluster i)")
    header = "    " + "group".ljust(8) + "".join(f"  c={i}".rjust(8) for i in range(k))
    print(header)
    for g in groups:
        m = y == g
        row = "    " + names[g].ljust(8)
        row += "".join(f"{p_c_z[m, k_idx, i].mean():8.3f}" for i in range(k))
        row += f"   (n={int(m.sum())})"
        print(row)


# Okabe-Ito: categorical, colourblind-safe. Only used to separate the true groups
# in the strip layout, where colour is an identity channel, not a magnitude one.
GROUP_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#56B4E9", "#D55E00", "#F0E442", "#999999"]


def plot_strip(p_c_z, k_idx, k, y, label_names, out_path, dpi,
               cluster_ids=None, display_ids=None):
    """One panel per cluster: p(c=i|k) for every cell, split by true group.

    Used when the run's UMAP coordinates are not available. The question this
    answers -- whether transitional cells get a graded posterior -- is about the
    distribution of probabilities within each group, which needs no embedding.
    """
    if y is None:
        raise SystemExit(
            "Without UMAP coordinates the posterior can only be shown against the "
            "true groups, but no labels were found. Pass --labels."
        )
    groups = np.unique(y)
    names = [label_names[j] if label_names and j < len(label_names) else str(g)
             for j, g in enumerate(groups)]

    cluster_ids = list(range(k)) if cluster_ids is None else list(cluster_ids)
    display_ids = list(cluster_ids) if display_ids is None else list(display_ids)
    n_panels = len(cluster_ids)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)

    for panel, i in enumerate(cluster_ids):
        ax = axes[panel]
        for j, g in enumerate(groups):
            v = p_c_z[y == g, k_idx, i]
            x = j + rng.uniform(-0.28, 0.28, size=v.shape[0])
            ax.scatter(x, v, s=2, alpha=0.25, linewidths=0,
                       color=GROUP_COLORS[j % len(GROUP_COLORS)], rasterized=True)
            # Median bar: the strip alone reads as a cloud at these point counts.
            ax.plot([j - 0.34, j + 0.34], [np.median(v)] * 2,
                    color="black", lw=1.8, solid_capstyle="butt", zorder=3)
        ax.set_title(f"p(c = {display_ids[panel]} | k = {k})", fontsize=TITLE_FS)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(names, fontsize=TICK_FS)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("0.7")
        if panel % ncols == 0:
            ax.set_ylabel("posterior probability", fontsize=LABEL_FS)

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def plot(p_c_z, k_idx, k, xy, y, label_names, out_path, dpi, cmap,
         cluster_ids=None, display_ids=None):
    # Smallest grid that fits; four panels give the 2x2.
    cluster_ids = list(range(k)) if cluster_ids is None else list(cluster_ids)
    display_ids = list(cluster_ids) if display_ids is None else list(display_ids)
    n_panels = len(cluster_ids)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    sm = None
    for panel, i in enumerate(cluster_ids):
        ax = axes[panel]
        p = p_c_z[:, k_idx, i]
        # Draw ascending in p so high-probability cells sit on top of the
        # low-probability field rather than being buried by it.
        order = np.argsort(p)
        sm = ax.scatter(
            xy[order, 0], xy[order, 1], c=p[order], s=3, cmap=cmap,
            vmin=0.0, vmax=1.0, linewidths=0, rasterized=True,
        )
        ax.set_title(f"p(c = {display_ids[panel]} | k = {k})", fontsize=TITLE_FS)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.85")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    # One shared 0..1 scale: panels are only comparable if the ramp does not
    # rescale per panel.
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.01)
    cbar.set_label("posterior probability", fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="run directory with p_c_z_best.npy")
    ap.add_argument("--k", type=int, default=None,
                    help="k to slice (default: truth_k if in range, else selected k)")
    ap.add_argument("--label-names", default=None,
                    help="comma-separated names for the true groups, e.g. d0,d2,d4,d7")
    ap.add_argument("--labels-file", default=None,
                    help="path to the label file the run was trained on "
                         "(default: labels_true.txt in the results dir)")
    ap.add_argument("--out-file", default=None, help="output path (default: <results-dir>/figS10.png)")
    ap.add_argument("--drop-empty", action="store_true",
                    help="omit components no cell is assigned to; at k=6 with two "
                         "empty components this gives the 2x2 of the real clusters")
    ap.add_argument("--relabel", action="store_true",
                    help="renumber the plotted clusters consecutively for display, "
                         "closing the gaps left by --drop-empty")
    ap.add_argument("--relabel-start", type=int, default=1,
                    help="first number to use with --relabel (default 1)")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--cmap", default=CMAP)
    args = ap.parse_args(argv)

    p_c_z, meta, xy, y = load_run(args.run_dir, args.labels_file)
    b = meta.get("b") or p_c_z.shape[2]
    a = meta.get("a")
    if a is None:
        # The k axis has one entry per k in [a, b] and the component axis is padded
        # to b, so a is determined exactly by the shape -- no guessing involved.
        a = p_c_z.shape[2] - p_c_z.shape[1] + 1
        print(f"no p_c_z_meta.json; inferred a={a}, b={b} from shape {p_c_z.shape}")

    k = args.k
    if k is None:
        if meta.get("truth_k_in_range", meta.get("truth_k") is not None) and meta.get("truth_k"):
            k = int(meta["truth_k"])
        else:
            k = int(meta.get("k_selected", a))
    if not (a <= k <= b):
        raise SystemExit(f"k={k} is outside the searched range [{a}, {b}].")
    k_idx = k - a

    label_names = args.label_names.split(",") if args.label_names else None

    print(f"posterior {p_c_z.shape}  k range [{a}, {b}]  plotting k={k} (index {k_idx})")
    if meta.get("k_selected") is not None and meta["k_selected"] != k:
        print(f"  note: the model selected k={meta['k_selected']}; plotting k={k}")
    check_floor(p_c_z, k_idx, k, b, meta)
    summarise_by_truth(p_c_z, k_idx, k, y, label_names)

    # A k-slice can leave components unused.
    hard = np.argmax(p_c_z[:, k_idx, :k], axis=1)
    occupied = [i for i in range(k) if (hard == i).sum() > 0]
    if len(occupied) < k:
        print(f"\n  {len(occupied)} of {k} components are occupied: {occupied} "
              f"(empty: {[i for i in range(k) if i not in occupied]})")
    cluster_ids = occupied if args.drop_empty else list(range(k))

    # Renumber the kept components consecutively for display; cluster_ids still
    # indexes the posterior.
    if args.relabel:
        display_ids = list(range(args.relabel_start, args.relabel_start + len(cluster_ids)))
        print("  relabelled for display: " +
              ", ".join(f"c={o} -> {n}" for o, n in zip(cluster_ids, display_ids)))
    else:
        display_ids = list(cluster_ids)

    out = args.out_file or os.path.join(args.run_dir, "figS10.png")
    if xy is not None:
        plot(p_c_z, k_idx, k, xy, y, label_names, out, args.dpi,
             resolve_cmap(args.cmap), cluster_ids, display_ids)
    else:
        print("\nno dmvae_umap_2d.npy for this run -- plotting p(c=i|k) against the "
              "true groups instead of on the embedding.")
        plot_strip(p_c_z, k_idx, k, y, label_names, out, args.dpi,
                   cluster_ids, display_ids)


if __name__ == "__main__":
    main()
