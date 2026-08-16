"""Figure 4 (d) -- Sankey, Mouse hypothalamus.

Called as a module by fig4cd.py; runnable on its own for the other methods.

Create Sankey plots for mouse_h: true cell types (left) vs predicted clusters (right).
Supports multiple methods: DMVAE, scVI, scGNN, ADClust, scAce, scDAC.

Uses data_label.txt for true labels (cell type names; no header, one label per line).
Requires: numpy, pandas, plotly. Optional: kaleido (for PNG output).

Examples:
  python fig4d_sankey.py --methods all
  python fig4d_sankey.py --methods dmvae,scvi,scgnn,adclust,scace,scdac
  python fig4d_sankey.py --methods dmvae --input-dir /path/to/mouse_h
"""

import argparse
import os
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    raise SystemExit("Install plotly: pip install plotly")

# Default paths
DIRECTORY = os.environ.get("DMVAE_DIRECTORY", ".")
DEFAULT_DATA_LABEL = os.environ.get(
    "MOUSE_H_LABELS", os.path.join(DIRECTORY, "Data", "mouse_h", "data_label.txt")
)
PAPER_DIR = os.environ.get(
    "FIGURE_OUTPUT_ROOT", os.path.join(DIRECTORY, "results", "dmvae", "figures")
)
INPUT_DIR_MOUSE_H = os.environ.get(
    "MOUSE_H_RESULTS_DIR", os.path.join(DIRECTORY, "results", "mouse_h")
)
DPI_PNG = 300

# Method key (npz filename) -> display name for title
METHODS = {
    "dmvae": "DMVAE",
    "scvi": "scVI",
    "scgnn": "scGNN",
    "adclust": "ADClust",
    "scace": "scAce",
    "scdac": "scDAC",
}


def load_y_true_from_data_label(label_path):
    """Load true labels from data_label.txt (no header, one cell type name per line)."""
    label_df = pd.read_csv(label_path, header=None, sep=None, engine="python", dtype=str)
    y_true = np.array([str(x).strip('"') for x in label_df.iloc[:, 0].values])
    return y_true


def _extract_1d_labels(arr):
    """Get 1D integer cluster labels from an array (unwrap object, flatten, squeeze)."""
    arr = np.asarray(arr)
    while arr.dtype == object and arr.size > 0:
        # Prefer first element for scVI-style (single run); else last
        arr = np.asarray(arr.flat[0] if arr.size == 1 else arr.flat[-1])
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    out = np.asarray(arr, dtype=int).squeeze()
    return np.atleast_1d(out)


def _debug_npz(data, npz_path, method_key):
    """Print structure of npz for debugging (--debug)."""
    print(f"[debug] {npz_path} (method={method_key})")
    print(f"  keys: {list(data.keys())}")
    for k in data.keys():
        a = data[k]
        try:
            a = np.asarray(a)
            print(f"  {k}: shape={a.shape}, dtype={a.dtype}, size={a.size}")
            if a.dtype == object and a.size <= 3:
                for i in range(a.size):
                    el = np.asarray(a.flat[i])
                    print(f"    [{i}]: shape={el.shape}, dtype={el.dtype}")
        except Exception as e:
            print(f"  {k}: {e}")


def load_y_pred_from_npz(npz_path, method_key=None, debug=False):
    """Load predicted clusters from npz; handle both flat and nested 'Clusters'.
    For scAce: use last run, last layer (Clusters[-1][-1]).
    For scVI: try 'Clusters' then 'clusters'/'y_pred'; unwrap or use per-cell object array.
    """
    data = np.load(npz_path, allow_pickle=True)
    if debug:
        _debug_npz(data, npz_path, method_key)
    # Try keys in order (some pipelines use lowercase or different name)
    clusters = None
    for key in ("Clusters", "clusters", "y_pred", "cluster", "assignments"):
        if key in data:
            clusters = np.asarray(data[key])
            break
    if clusters is None:
        raise KeyError("No Clusters/clusters/y_pred in npz; keys: " + str(list(data.keys())))

    expected_n = None
    if "Embedding" in data:
        emb = np.asarray(data["Embedding"])
        if emb.ndim >= 1 and emb.size > 0:
            expected_n = emb.shape[0] if emb.ndim >= 1 else int(np.size(emb))

    # scAce: Clusters is nested (e.g. runs x layers x n_cells); take last run then last layer
    if method_key == "scace":
        c = clusters
        while c.dtype == object and c.size > 0:
            c = np.asarray(c.flat[-1])
        if c.ndim > 1:
            c = c[-1]  # last layer -> 1D (n_cells,)
        y_pred = np.asarray(c, dtype=int).squeeze()
    else:
        # Default: unwrap object / flatten to 1D
        y_pred = _extract_1d_labels(clusters)
        y_pred = np.atleast_1d(y_pred)
        n_cells = int(np.size(y_pred))
        if expected_n is None:
            expected_n = n_cells

        # scVI / others: if only 1 cluster, try other interpretations
        if expected_n > 1 and len(np.unique(y_pred)) == 1:
            # (1) (n_cells,) object: one label per cell (e.g. 0-d int per element)
            if clusters.dtype == object and clusters.ndim == 1 and clusters.size == expected_n:
                try:
                    out = np.array([int(np.asarray(c).flat[0]) for c in clusters.flat], dtype=np.int64)
                    if len(np.unique(out)) > 1:
                        y_pred = out
                except Exception:
                    pass
            # (2) (1,) or () object: try first and last element as (n_cells,) array
            if len(np.unique(y_pred)) == 1 and clusters.dtype == object and clusters.size > 0:
                for idx in (0, -1):
                    alt = np.asarray(clusters.flat[idx])
                    if np.size(alt) == expected_n:
                        alt = np.asarray(alt, dtype=int).reshape(-1)
                        alt = np.atleast_1d(alt)
                        if len(np.unique(alt)) > 1:
                            y_pred = alt
                            break
            # (3) Other keys in npz with length expected_n and >1 unique
            if len(np.unique(y_pred)) == 1:
                for key in list(data.keys()):
                    if key.lower() in ("clusters", "y_pred", "cluster", "assignments", "embedding"):
                        continue
                    try:
                        arr = np.asarray(data[key])
                        if arr.ndim == 1 and arr.size == expected_n:
                            arr = np.asarray(arr, dtype=int).squeeze()
                            arr = np.atleast_1d(arr)
                            if len(np.unique(arr)) > 1:
                                y_pred = arr
                                break
                        elif arr.ndim == 2 and arr.shape[0] == expected_n and min(arr.shape) == 1:
                            arr = np.asarray(arr.reshape(-1), dtype=int).squeeze()
                            arr = np.atleast_1d(arr)
                            if len(np.unique(arr)) > 1:
                                y_pred = arr
                                break
                    except Exception:
                        continue
    if y_pred.ndim == 0:
        y_pred = np.atleast_1d(y_pred)
    return y_pred


def build_sankey_fig(y_true, y_pred, method_label, method_key=None):
    """Build one Sankey figure: true cell types (from data_label) -> predicted clusters."""
    n = min(len(y_pred), len(y_true))
    y_true_n = y_true[:n]
    y_pred_n = y_pred[:n]

    unique_true = sorted(np.unique(y_true_n))
    unique_pred = sorted(np.unique(y_pred_n))

    # y_true are already cell type names (from data_label.txt)
    left_labels = [str(c) for c in unique_true]
    right_labels = [f"Pred_{c}" for c in unique_pred]

    n_left = len(unique_true)
    # For DMVAE only: show text labels only for the specific cell types below.
    # All other left nodes still appear in the sankey, just without a text label.
    dmvae_label_whitelist = {
        "MO", "POPC", "Macro", "Micro", "OPC", "Astro",
        "GABA9", "GABA8", "SCO", "Ependy", "Tany",
        "Endo1", "Endo2", "IMO", "Glu7", "Glu5",
    }
    if method_key == "dmvae":
        left_labels_display = [
            lbl if lbl in dmvae_label_whitelist else "" for lbl in left_labels
        ]
    else:
        left_labels_display = list(left_labels)

    node_labels = left_labels_display + right_labels
    true_to_idx = {c: i for i, c in enumerate(unique_true)}
    pred_to_idx = {c: n_left + i for i, c in enumerate(unique_pred)}

    source, target, value = [], [], []
    for gt in unique_true:
        mask_gt = y_true_n == gt
        for pr in unique_pred:
            count = np.sum(mask_gt & (y_pred_n == pr))
            if count > 0:
                source.append(true_to_idx[gt])
                target.append(pred_to_idx[pr])
                value.append(int(count))

    # One hue per annotated cell type; each predicted cluster takes the hue of the
    # type it corresponds to. Imported lazily -- fig4cd imports this module.
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from fig4cd import true_colors, matched_pred_colors

        node_colors = true_colors(unique_true)
        node_colors += matched_pred_colors(y_pred_n, y_true_n,
                                           [str(c) for c in unique_pred])
    except Exception as exc:      # standalone use without fig4cd.py alongside
        print(f"  [warn] falling back to the local palette ({exc})")
        left_colors = [
            "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
            "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
            "#ccebc5", "#ffed6f", "#a6cee3", "#1f78b4", "#b2df8a",
        ]
        pred_palette = ["#e0e0e0", "#ffcccc", "#ccffcc", "#ccccff",
                        "#ffffcc", "#ffccff", "#ccffff"]
        node_colors = [left_colors[i % len(left_colors)] for i in range(n_left)]
        node_colors += [pred_palette[i % len(pred_palette)]
                        for i in range(len(unique_pred))]

    def hex_to_rgba(hex_color, alpha=0.5):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    link_colors = [hex_to_rgba(node_colors[s], 0.5) for s in source]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(source=source, target=target, value=value, color=link_colors),
    )])

    if method_key == "dmvae":
        # Figure 4d: narrow and tall so it pairs with the panel (c) UMAP stack
        # (7 x 11 in) alongside it -- a little under twice that height.
        fig.update_layout(
            title=None,
            font=dict(size=30),
            height=1500,
            width=650,
            margin=dict(l=60, r=60, t=20, b=40),
        )
    else:
        fig.update_layout(
            title=dict(
                text=f"mouse_h {method_label}: True cell types → Predicted clusters",
                font=dict(size=16), x=0.5, xanchor="center",
            ),
            font=dict(size=12),
            height=700,
            width=900,
            margin=dict(l=80, r=80, t=60, b=40),
        )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sankey: true cell types vs predicted clusters (mouse_h) for one or more methods.",
    )
    parser.add_argument("--data-label", default=DEFAULT_DATA_LABEL, help="Path to data_label.txt (true cell type names; no header)")
    parser.add_argument("--data-dir", default=INPUT_DIR_MOUSE_H, help="Directory containing {method}.npz files")
    parser.add_argument("--out-dir", default=PAPER_DIR, help="Directory for output HTML and PNG")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method keys or 'all' (e.g. dmvae,scvi,scgnn,adclust,scace,scdac)",
    )
    parser.add_argument("--dpi", type=int, default=DPI_PNG, help="PNG resolution in PPI (default 300)")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG output (only save HTML)")
    parser.add_argument("--debug", action="store_true", help="Print npz structure for each method (shape/dtype of arrays)")
    args = parser.parse_args(argv)

    data_label = os.path.abspath(args.data_label)
    input_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.out_dir)

    if not os.path.isfile(data_label):
        raise SystemExit(f"Data file not found: {data_label}")
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.methods.strip().lower() == "all":
        method_keys = list(METHODS.keys())
    else:
        method_keys = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
        for m in method_keys:
            if m not in METHODS:
                raise SystemExit(f"Unknown method: {m}. Choose from: {', '.join(METHODS.keys())}")

    # Load true labels from data_label.txt (cell type names; no header)
    y_true = load_y_true_from_data_label(data_label)

    for method_key in method_keys:
        method_label = METHODS[method_key]
        npz_path = os.path.join(input_dir, f"{method_key}.npz")
        if not os.path.isfile(npz_path):
            print(f"Skipping {method_label}: not found {npz_path}")
            continue

        try:
            y_pred = load_y_pred_from_npz(npz_path, method_key=method_key, debug=args.debug)
        except Exception as e:
            print(f"Skipping {method_label}: failed to load Clusters from {npz_path}: {e}")
            continue

        # Drop empty clusters and renumber the remaining cluster IDs contiguously from 0
        present = np.unique(y_pred)
        n_total = int(present.max()) + 1 if present.size else 0
        if len(present) < n_total:
            print(
                f"  {method_label}: {n_total - len(present)} empty cluster(s) removed; "
                f"{len(present)} non-empty cluster(s) remain"
            )
        remap = {int(old): new for new, old in enumerate(present)}
        y_pred = np.array([remap[int(v)] for v in y_pred], dtype=int)

        fig = build_sankey_fig(y_true, y_pred, method_label, method_key=method_key)

        base = f"sankey_mouse_h_{method_key}"
        out_html = os.path.join(output_dir, f"{base}.html")
        out_png = None if args.no_png else os.path.join(output_dir, f"{base}.png")

        pio.write_html(fig, file=out_html, include_plotlyjs=True)
        print(f"  {method_label}: Saved HTML {out_html}")

        if out_png:
            try:
                scale = args.dpi / 96.0
                # Take the size from the figure itself; the DMVAE panel is a tall,
                # narrow variant and a hardcoded size here would override it.
                pio.write_image(fig, out_png,
                                width=fig.layout.width or 900,
                                height=fig.layout.height or 700,
                                scale=scale)
                print(f"  {method_label}: Saved PNG ({args.dpi} PPI) {out_png}")
            except Exception as e:
                print(f"  {method_label}: PNG save failed (install kaleido): {e}")

    print("Done. Open the HTML files in a browser to view the Sankey diagrams.")


if __name__ == "__main__":
    main()
