"""Figure 3 (c, d) -- UMAP grids for the three main-text datasets.

One row per dataset (Plasschaert, Turtle brain, Mouse hypothalamus), one column
per method. Panel (c) colours cells by each method's own predicted cluster;
panel (d) colours the same coordinates by the curated cell-type label.

The comparison methods are read from ``Results/<dir>/<method>.npz`` and DMVAE from
``best_ae_realworld/<dataset>/dmvae.npz``.

UMAP coordinates are expensive and are cached per dataset next to the results as
``umap_<tag>.npz``; delete that file (or pass --recompute-umap) to rebuild them.

    python fig3cd.py
    python fig3cd.py --panel c --out-dir /path/out
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as _ARI

plt.ioff()

DIRECTORY = os.environ.get("DMVAE_DIRECTORY", ".")
OUTPUT_DIR = os.environ.get(
    "FIGURE_OUTPUT_ROOT", os.path.join(DIRECTORY, "results", "dmvae", "figures")
)
RESULTS_ROOT = os.environ.get(
    "REALWORLD_RESULTS_ROOT", os.path.join(DIRECTORY, "results")
)
BEST_AE_ROOT = os.environ.get(
    "DMVAE_BEST_AE_ROOT", os.path.join(RESULTS_ROOT, "best_ae_realworld")
)
DATA_ROOT = os.environ.get("REALWORLD_DATA_ROOT", os.path.join(DIRECTORY, "Data"))

METHODS = ["scVI", "scGNN", "ADClust", "scAce", "scDAC", "DMVAE"]
NPZ_NAME = {"scVI": "scvi", "scGNN": "scgnn", "ADClust": "adclust",
            "scAce": "scace", "scDAC": "scdac"}

# Methods whose npz already carries the score computed against the curated
# labels; the rest are scored here from the assignment.
TRUST_STORED_SCORE = {"scVI", "scGNN", "scDAC", "DMVAE"}

TITLE_FS = 30
DPI = 300
PANEL_W, PANEL_H = 5.0, 4.0

# results_dir: comparison npz; best_ae_dir: DMVAE; data_dir: curated labels.
# labels / extra_labels: curated vectors to try, ("h5", "cell_type1") by default;
# methods were not all run on the same cell subset, so more than one may be
# needed and resolve_labels checks which fits.
# dmvae_source picks which DMVAE run the row is drawn from: "best_ae" is the
# grid_20260801 winner, "results" is the earlier run kept for this figure.
# frozen: methods whose coordinates must be taken from the cache as-is and never
# recomputed, because another figure draws the same points. fig4cd.py reads
# umap_mouseh.npz["UMAP"]["DMVAE"] for Figure 4c, so recomputing it here would
# move that panel too; Figure 3c and Figure 4c have to show the same UMAP.
DATASETS = [
    dict(name="Plasschaert", results_dir="Plass", best_ae_dir="Plasschaert",
         data_dir="Plasschaert", dmvae_source="best_ae", frozen=(),
         umap_tag="plass"),
    dict(name="Turtle brain", results_dir="turtle_b", best_ae_dir="turtle_b",
         data_dir="turtle_b", dmvae_source="best_ae", frozen=(),
         umap_tag="turtleb"),
    dict(name="Mouse hypothalamus", results_dir="mouse_h", best_ae_dir="mouse_h",
         data_dir="mouse_h", labels=("txt", "data_celltype.txt"),
         extra_labels={"data.h5 cell_type1": ("h5", "cell_type1")},
         dmvae_source="results", frozen=("DMVAE",),
         # This panel is Figure 4c: draw it with fig4cd.py's own cluster
         # renumbering and palette, off the same cell-type names it uses.
         fig4_style=("DMVAE",), fig4_label_file="data_label.txt",
         umap_tag="mouseh"),
]


def _encode(values) -> np.ndarray:
    """Labels of any dtype -> contiguous integer codes."""
    return np.unique(np.asarray(values).astype(str), return_inverse=True)[1]


def _read_label_source(cfg: dict, source) -> np.ndarray:
    """One curated label vector, from data.h5 or a text file beside it."""
    kind, key = source
    if kind == "h5":
        with h5py.File(os.path.join(DATA_ROOT, cfg["data_dir"], "data.h5"), "r") as h:
            return _encode(np.array(h[key] if key == "Y" else h["obs"][key]))
    if kind == "txt":
        return _encode(np.loadtxt(os.path.join(DATA_ROOT, cfg["data_dir"], key)))
    raise ValueError(f"unknown label source {source!r}")


def load_labels(cfg: dict) -> dict:
    """Every curated label vector this dataset offers, keyed for reporting.

    Methods were not all run on the same cell subset, so there is no single
    right answer here; resolve_labels picks per method and checks its choice.
    """
    out = {"curated": _read_label_source(cfg, cfg.get("labels", ("h5", "cell_type1")))}
    for name, source in (cfg.get("extra_labels") or {}).items():
        out[name] = _read_label_source(cfg, source)
    return out


def resolve_labels(method: str, res: dict, curated: dict, others: dict,
                   tol: float = 0.02):
    """The label vector that actually lines up with this method's assignment.

    Methods that filtered cells cannot be scored against the shared curated
    vector by position, so each candidate is checked against the method's own
    stored ARI and the first that reproduces it wins. Order of preference: the
    method's own stored Labels, then the curated vectors, then any other
    method's Labels -- the last covers a filtered subset with no curated vector
    of its own, as on PBMC, where scGNN's barcode-matched Labels turn out to be
    the truth for all five comparison methods.
    """
    clusters = res["clusters"]
    stored = res["ari"]

    candidates = []
    if res["labels"] is not None:
        candidates.append((f"{method}.npz Labels", res["labels"]))
    candidates.extend(curated.items())
    for name, other in others.items():
        if name != method and other["labels"] is not None:
            candidates.append((f"{name}.npz Labels", other["labels"]))

    if stored is None:
        return candidates[0][1], candidates[0][0]

    best = None
    for name, vec in candidates:
        n = min(len(vec), len(clusters))
        gap = abs(_ARI(np.asarray(vec)[:n], clusters[:n]) - stored)
        if gap <= tol:
            return vec, name
        if best is None or gap < best[0]:
            best = (gap, name, vec)

    print(f"    [{method}] no label vector reproduces the stored ARI "
          f"({stored:.4f}); closest is {best[1]} off by {best[0]:.4f}")
    return best[2], best[1] + " (unverified)"


def load_method(cfg: dict, method: str) -> dict:
    """Embedding, hard assignment, stored scores and own labels for one method."""
    if method == "DMVAE":
        root = (RESULTS_ROOT if cfg.get("dmvae_source") == "results"
                else BEST_AE_ROOT)
        sub = (cfg["results_dir"] if cfg.get("dmvae_source") == "results"
               else cfg["best_ae_dir"])
        path = os.path.join(root, sub, "dmvae.npz")
    else:
        path = os.path.join(RESULTS_ROOT, cfg["results_dir"],
                            f"{NPZ_NAME[method]}.npz")

    d = np.load(path, allow_pickle=True)
    emb = np.asarray(d["Embedding"])
    if emb.ndim == 3:                       # ADClust / scAce keep every round
        emb = emb[-1]

    clusters = d["Clusters"]
    if method == "scAce":                   # (round, iteration, cell)
        clusters = np.asarray(clusters.tolist() if clusters.ndim == 0
                              else clusters)[-1]
        if np.asarray(clusters).ndim == 2:
            clusters = np.asarray(clusters)[-1]
    clusters = np.asarray(clusters).ravel().astype(int)

    def stored(key):
        if key not in d.files:
            return None
        arr = np.asarray(d[key], dtype=float).ravel()
        return float(arr[-1]) if arr.size else None

    # A method that filtered cells may store the labels it was actually scored
    # against -- scgnn_eval.py matches them back to the barcode column of
    # scgnn_celltype.txt. They are only a candidate here; resolve_labels decides
    # whether they line up, since not every archive's Labels are in assignment
    # order (scAce's on mouse_h are not).
    labels = (np.asarray(d["Labels"]).ravel().astype(int)
              if "Labels" in d.files else None)

    return dict(emb=emb, clusters=clusters, labels=labels,
                ari=stored("ARI"), nmi=stored("NMI"), path=path)


def _source_stamp(loaded: dict) -> dict:
    """Size and mtime of every source archive, so a re-run invalidates the cache."""
    return {m: (os.path.getsize(r["path"]), int(os.path.getmtime(r["path"])))
            for m, r in loaded.items()}


def umap_coords(cfg: dict, loaded: dict, recompute: bool) -> dict:
    """UMAP per method, cached alongside the results.

    The cache carries a stamp of the archives it was built from, so swapping in
    a new dmvae.npz rebuilds the coordinates instead of silently plotting the
    previous run's embedding. Methods listed in cfg["frozen"] are exempt: they
    are always taken from the cache, because another figure plots the same
    points and must not move when this one is redrawn.
    """
    cache = os.path.join(RESULTS_ROOT, f"umap_{cfg['umap_tag']}.npz")
    frozen = tuple(cfg.get("frozen", ()))
    stamp = _source_stamp(loaded)

    cached_coords = {}
    if os.path.isfile(cache):
        cached = np.load(cache, allow_pickle=True)
        cached_coords = cached["UMAP"].item()
        old = cached["SOURCES"].item() if "SOURCES" in cached.files else None
        fresh = all(m in cached_coords for m in METHODS) and old == stamp
        if fresh and not recompute:
            print(f"  UMAP from cache {cache}")
            return cached_coords

    missing = [m for m in frozen if m not in cached_coords]
    if missing:
        raise FileNotFoundError(
            f"{cfg['name']}: {', '.join(missing)} coordinates are frozen but not "
            f"in {cache}. They are shared with another figure, so they cannot be "
            f"regenerated here -- restore the cache first."
        )

    coords = {}
    for method in METHODS:
        if method in frozen:
            coords[method] = np.asarray(cached_coords[method])
            print(f"  UMAP {method}: {coords[method].shape}  (frozen, from cache)")
            continue
        adata = sc.AnnData(np.asarray(loaded[method]["emb"], dtype=np.float32))
        sc.pp.neighbors(adata)
        sc.tl.umap(adata, random_state=0)
        coords[method] = np.asarray(adata.obsm["X_umap"])
        print(f"  UMAP {method}: {coords[method].shape}")

    np.savez(cache, UMAP=coords, SOURCES=stamp)
    print(f"  Cached UMAP -> {cache}")
    return coords


def _style(ax) -> None:
    """Notebook panel styling: no ticks, only the left and bottom rules."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.plot([xmin, xmax], [ymin, ymin], color="black", linewidth=1)
    ax.plot([xmin, xmin], [ymin, ymax], color="black", linewidth=1)
    ax.set_facecolor("white")


def plot_panel(ax, method: str, res: dict, coords: np.ndarray,
               y_true: np.ndarray, by: str, fig4_names: np.ndarray | None = None) -> None:
    """One UMAP panel.

    fig4_names carries the curated cell-type NAMES when this panel also appears
    in Figure 4c. It switches the panel to fig4cd.py's exact drawing: empty
    clusters dropped and renumbered from 0, predicted clusters coloured by the
    cell type they best match, cell types coloured by fig4's palette. Without it
    the points would carry scanpy's default colours against raw cluster ids, and
    the two figures would show the same coordinates in different colours.
    """
    y_pred = res["clusters"]
    n = min(y_pred.size, y_true.size, len(coords))
    if fig4_names is not None:
        n = min(n, len(fig4_names))
    y_pred, y_use, uu = y_pred[:n], y_true[:n], coords[:n]

    if by == "pred":
        if method in TRUST_STORED_SCORE and res["ari"] is not None:
            ari = res["ari"]
        else:
            ari = metrics.adjusted_rand_score(y_use, y_pred)
        title = f"K = {np.unique(y_pred).size}   ARI = {ari:.2f}"
    else:
        title = f"K = {np.unique(y_use).size}"

    adata = sc.AnnData(pd.DataFrame(np.zeros((n, 1))))
    adata.obsm["X_umap"] = uu

    if fig4_names is None:
        colour = y_pred if by == "pred" else y_use
        adata.obs["c"] = pd.Categorical(np.asarray(colour).astype(str))
    else:
        import fig4cd

        names = np.asarray(fig4_names)[:n]
        if by == "pred":
            present = np.unique(y_pred)                  # drop empty clusters
            remap = {int(old): new for new, old in enumerate(present)}
            renumbered = np.array([remap[int(v)] for v in y_pred], dtype=int)
            cats = [str(i) for i in range(len(present))]
            adata.obs["c"] = pd.Categorical(renumbered.astype(str), categories=cats)
            adata.uns["c_colors"] = fig4cd.matched_pred_colors(renumbered, names, cats)
        else:
            adata.obs["c"] = pd.Categorical(names)
            adata.uns["c_colors"] = fig4cd.true_colors(adata.obs["c"].cat.categories)

    sc.pl.umap(adata, color=["c"], ax=ax, show=False, legend_loc=None, size=8)
    ax.set_title(title, fontsize=TITLE_FS, family="Arial")
    _style(ax)


def make_grid(out_path: str, by: str, recompute_umap: bool,
              datasets: list | None = None) -> None:
    """One row per dataset, one column per method. Shared with Supplementary S3/S4."""
    datasets = datasets if datasets is not None else DATASETS

    fig, axs = plt.subplots(len(datasets), len(METHODS),
                            figsize=(PANEL_W * len(METHODS),
                                     PANEL_H * len(datasets)),
                            dpi=DPI)
    axs = np.atleast_2d(axs)

    for row, cfg in enumerate(datasets):
        print(f"{cfg['name']}  ({by})")
        curated = load_labels(cfg)
        loaded = {m: load_method(cfg, m) for m in METHODS}
        coords = umap_coords(cfg, loaded, recompute_umap)

        fig4_names = None
        if cfg.get("fig4_label_file"):
            path = os.path.join(DATA_ROOT, cfg["data_dir"], cfg["fig4_label_file"])
            df = pd.read_csv(path, header=None, sep=None, engine="python", dtype=str)
            fig4_names = np.array([str(x).strip('"') for x in df.iloc[:, 0].values])

        for col, method in enumerate(METHODS):
            labels, source = resolve_labels(method, loaded[method], curated, loaded)
            shared = method in tuple(cfg.get("fig4_style", ()))
            plot_panel(axs[row][col], method, loaded[method], coords[method],
                       labels, by, fig4_names if shared else None)
            print(f"    {method:<8} {axs[row][col].get_title():<22} "
                  f"labels: {source}{'   [fig4c colours]' if shared else ''}")

    fig.savefig(out_path, dpi=DPI, format="png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render Figure 3 panels c and d.")
    p.add_argument("--panel", choices=["c", "d", "all"], default="all")
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--recompute-umap", action="store_true",
                   help="ignore the cached coordinates and rebuild them")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.panel in ("c", "all"):
        make_grid(f"{args.out_dir}/rw.png", "pred", args.recompute_umap)
    if args.panel in ("d", "all"):
        make_grid(f"{args.out_dir}/umap_truth.png", "true", args.recompute_umap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
