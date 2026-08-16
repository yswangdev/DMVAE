"""Supplementary Figure S7 -- multiresolution UMAPs on Mouse hypothalamus.

Rows are scVI, scGNN, ADClust, scAce, and DMVAE. The first three columns
show low, default/selected, and high resolutions; the last column shows the
curated labels. This file is synchronized from DMVAE-run's resolution-sweep
plotter and remains usable with another dataset through command-line arguments.

For every comparison method that exposes a genuine cluster-number knob, the whole
algorithm was re-run at a LOW and a HIGH setting (see
``scripts/slurm/mouse_h_resolution_sweep.slurm.sh``). This script draws one row per
method and, per setting, the method's OWN embedding UMAP-projected and coloured by its
predicted clusters -- and, optionally, the same layout coloured by the true labels.

Layout, with --umap_cache:

* DMVAE -- every column uses the cached projection. Its per-k assignments all come from
  ONE trained model over ONE latent space, so a single layout is the correct depiction
  and the row matches the published panel exactly.
* the four comparison methods -- only the DEFAULT column (and the truth panel, which
  reuses the default run's layout) uses the cached projection, because that is the run
  the cache was computed from. Every other value RE-TRAINS the method, producing a
  genuinely different embedding, so those columns are projected on their own.

Without --umap_cache every run is projected separately.

Comparison methods only -- DMVAE is not plotted here.

scDAC is absent by construction: its k comes from a Dirichlet-process mixture with
``n_components`` / ``weight_concentration_prior`` hardcoded in its ``run.py``, so it has
no resolution-style argument to sweep.

Usage
-----
    python figS7.py \
        --scenario mouse_h --paper_label "Mouse hypothalamus" \
        --res_root /path/to/mouse_h_res_sweep --out_dir /path/to/output \
        --dmvae_dir /path/to/dmvae/mouse_h --dmvae_labels /path/to/data_label.txt
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

try:
    from dmvae_run.training.utils import get_colors_cmap
except ImportError:
    # DMVAE-paper keeps the same helper under model/ rather than the installed
    # dmvae_run package used by the training repository.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from model.utils import get_colors_cmap


def _install_numpy2_pickle_shim():
    """Let numpy 1.x read object arrays pickled by numpy 2.x.

    The comparison methods run in their own conda envs, so a .npz written under numpy 2
    may well be read here under numpy 1. numpy 2 pickles object arrays with references to
    ``numpy._core.*``, which does not exist in numpy 1, giving
    ``ModuleNotFoundError: No module named 'numpy._core'`` at load time. Aliasing the few
    submodules the pickles actually name onto their numpy 1 ``numpy.core`` equivalents
    makes those files readable. No-op when already running numpy 2.
    """
    import importlib
    import sys

    if hasattr(np, "_core"):
        return
    try:
        import numpy.core as _np_core
    except ImportError:
        return
    sys.modules.setdefault("numpy._core", _np_core)
    for name in ("multiarray", "umath", "numeric", "numerictypes", "_multiarray_umath", "overrides"):
        try:
            sys.modules.setdefault(f"numpy._core.{name}", importlib.import_module(f"numpy.core.{name}"))
        except ImportError:
            continue


_install_numpy2_pickle_shim()

# --- Which knob each method exposes. The VALUES are CLI arguments (--scvi_res etc.) so
# --- they can be retuned without editing this file; the defaults below mirror the
# --- METHODS/VALUES/TAGS arrays in resolution_sweep.slurm.sh. A tag is simply
# --- <prefix><value>, so whatever you pass here must be spelled exactly as it was passed
# --- to the sweep ("0.85", not "0.850") for the filename to match.
METHOD_SPEC = [
    # (row label, results subdir, param name, the method's own default, tag prefix, CLI dest)
    ("scVI",    "scvi",    "resolution",          "1.0", "res", "scvi_res"),
    ("scGNN",   "scgnn",   "resolution",          "0.5", "res", "scgnn_res"),
    ("ADClust", "adclust", "dip_merge_threshold", "0.9", "dip", "adclust_dip"),
    ("scAce",   "scace",   "resolution",          "2",   "res", "scace_res"),
]

# Values are kept as STRINGS, never floats: the tag must reproduce the filename the sweep
# wrote, and float formatting would turn "4.0" into "4.0" but "0.80" into "0.8".
#
# These pairs are spread WIDE ENOUGH TO MOVE k. The first sweep used values close to each
# method's default and k barely budged: scVI 0.8/1.2 and scAce 1.5/2.5 both returned k=9,
# and ADClust 0.85/0.95 returned bit-identical results because no Dip score fell in that
# gap. See the header of resolution_sweep.slurm.sh for the per-method reasoning.
# THREE values per method: lower, "default", higher. "default" is a SENTINEL matching the
# sweep script's: that run passed NO parameter flag, so it is the method's untouched
# behaviour, and its file is tagged plain "default" rather than "<prefix>default".
DEFAULT_VALUES = {
    "scvi_res": ["0.1", "default", "3.0"],        # scVI leiden default 1.0
    "scgnn_res": ["0.2", "0.5", "2.0"],           # 0.5 IS scGNN's 'auto' default here
    "adclust_dip": ["0.5", "default", "0.99"],    # ADClust default 0.9
    "scace_res": ["0.3", "default", "5.0"],       # scAce default 2
}

# True k is read from each run's stored ``Labels`` rather than hardcoded: the data a
# sweep was actually run on need not match the generator currently in the repo (the
# committed s08_mixed_hard makes 9 groups of 1000, s04_many_equal 8 groups of 1000).
# Override with --truth_k only if a run stores no labels.
truth_k_fallback = 9   # s08_mixed_hard; only used when a run stores no Labels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res_root", required=True, help="Root holding <method>/results_sim_1__<tag>.npz")
    p.add_argument("--rep", type=int, default=1, help="Replicate index (rep 1 by default)")
    p.add_argument("--scenario", type=str, default="mouse_h",
                   help="Dataset/scenario name used in the title (default: mouse_h).")
    p.add_argument("--paper_label", type=str, default="Mouse hypothalamus",
                   help="How the manuscript refers to this scenario; shown alongside the "
                        "repo name. Pass an empty string to omit.")
    p.add_argument("--truth_k", type=int, default=None,
                   help="True cluster count. Default: read from each run's stored Labels.")
    p.add_argument("--out_dir", required=True)

    # One argument per method's knob. Any number of values is accepted -- the grid grows
    # to the widest row -- so a method can be shown at 2, 3 or more settings.
    for label, _sub, pname, _default, prefix, dest in METHOD_SPEC:
        p.add_argument(
            f"--{dest}", nargs="+", type=str, default=DEFAULT_VALUES[dest], metavar="V",
            help=f"{label} {pname} values to plot (tags '{prefix}<value>'). "
                 f"Default: {' '.join(DEFAULT_VALUES[dest])}",
        )

    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no_truth", action="store_true", help="Draw only predicted-label panels")
    p.add_argument("--drop_truth", nargs="*", default=[], metavar="METHOD",
                   help="Row labels whose stored ground truth is known to be unusable "
                        "(e.g. scGNN when cell names did not encode a unique index). "
                        "Their predictions are still drawn; ARI/NMI and the truth panel "
                        "are suppressed rather than shown wrong.")
    p.add_argument("--fig3c_dir", default=None, metavar="DIR",
                   help="Take the DEFAULT column's cluster assignment for the four "
                        "comparison methods from DIR/{scvi,scgnn,adclust,scace}.npz -- the "
                        "per-method archives figure 3c draws from -- instead of the sweep's "
                        "own default run. Makes that column identical to the published "
                        "panel. Non-default columns and the DMVAE row are unaffected.")
    p.add_argument("--umap_cache", default=None, metavar="NPZ",
                   help="Precomputed UMAP coordinates, as written for the manuscript "
                        "figures (npz with a 'UMAP' dict: method -> (n_cells, 2)). When a "
                        "method is present and its cell count matches, EVERY panel in that "
                        "row is drawn on those coordinates, so the layout matches the "
                        "published panel and columns differ only in colour. Rows without a "
                        "usable entry fall back to projecting each run separately.")
    p.add_argument("--dmvae_dir", default=None, metavar="DIR",
                   help="Add a DMVAE row from DIR/dmvae.npz + DIR/assignments_all_k.json. "
                        "Unlike the other rows this is ONE latent space recoloured per k, "
                        "since DMVAE's per-k assignments all come from the same model.")
    p.add_argument("--dmvae_k", nargs="+", default=["6", "10", "selected"], metavar="K",
                   help="DMVAE columns. An integer reads assignments_all_k.json; "
                        "'selected' uses dmvae.npz Clusters -- the model's own choice, "
                        "i.e. the clustering shown in figure 4c. Default: 6 10 selected")
    p.add_argument("--dmvae_labels", default=None, metavar="FILE",
                   help="Ground-truth labels for the DMVAE row (e.g. data_label.txt). "
                        "Falls back to a Labels key inside dmvae.npz.")
    p.add_argument("--point_size", type=float, default=1.5)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _to_2d_embedding(emb, n_expected, src):
    """Reduce a stage-wise embedding stack to the single 2-D embedding UMAP needs.

    ADClust (the scAce-bundled fork, which returns 7 values rather than upstream's 2)
    hands back ``embedded`` as a LIST of embeddings -- one at initialisation, one after
    training -- so it loads as ``(n_stages, n_cells, n_dims)``. Prefer the LAST stage
    whose row count matches the predictions, i.e. the trained embedding.
    """
    emb = np.asarray(emb)
    if emb.ndim == 2:
        return emb
    if emb.ndim == 3:
        matching = [i for i in range(emb.shape[0]) if emb[i].shape[0] == n_expected]
        i = matching[-1] if matching else emb.shape[0] - 1
        print(f"[INFO] {os.path.basename(src)}: stage-wise embedding {emb.shape}; using stage {i}")
        return np.asarray(emb[i])
    raise ValueError(f"{src}: embedding has unsupported shape {emb.shape}")


def _final_labels(arr):
    """Collapse a possibly nested per-stage label history to the final 1-D assignment.

    scAce's ``pred_all`` holds one entry per phase, and each entry is either a single
    label vector ``(n_cells,)`` or a stack of per-iteration vectors ``(n_iters, n_cells)``.
    The final clustering is therefore the LAST ROW of the LAST entry -- ravelling the
    whole thing would produce n_iters * n_cells labels. Plain 1-D arrays (ADClust,
    scGNN) and object arrays of strings (scVI's Leiden names) pass through untouched.
    """
    a = arr
    if isinstance(a, np.ndarray) and a.dtype == object and a.ndim == 1 and a.size:
        if isinstance(a.flat[0], (np.ndarray, list)):  # container of per-phase histories
            a = list(a)[-1]
    a = np.asarray(a)
    while a.ndim > 1:
        a = a[-1]
    return a


def _last_stage(arr):
    """scAce with return_all=True stores one entry PER STAGE; take the final one."""
    a = np.asarray(arr, dtype=object)
    if a.ndim == 0:
        a = a.item()
    if isinstance(a, (list, tuple)) or (isinstance(a, np.ndarray) and a.dtype == object):
        return np.asarray(list(a)[-1])
    return np.asarray(a)


def load_run(npz_path, method_dir):
    """Return (embedding, predicted labels, true labels) or None if the run is missing."""
    if not os.path.isfile(npz_path):
        print(f"[MISS] {npz_path}")
        return None
    d = np.load(npz_path, allow_pickle=True)

    def _get(plain_key, object_key, unwrap):
        """Prefer the plain-dtype key; fall back to the object-dtype one.

        Newer runs store final-stage results as ordinary arrays, which need no pickling
        and so are immune to numpy-version skew across conda envs. Older .npz files only
        have the object-dtype keys, which do get pickled.
        """
        if plain_key in d.files:
            return np.asarray(d[plain_key])
        try:
            raw = d[object_key]
        except ModuleNotFoundError as exc:  # numpy 2 pickle vs numpy 1 reader
            raise RuntimeError(
                f"{npz_path}: '{object_key}' was pickled by an incompatible numpy "
                f"({exc}). Re-run this method so it also writes the plain-dtype "
                f"'{plain_key}', or load with the numpy version that wrote the file."
            ) from exc
        return unwrap(raw)

    if method_dir == "scace":
        emb = _get("Embedding_final", "Embedding", _last_stage)
        pred = _get("Clusters_final", "Clusters", _final_labels)
    else:
        emb = _get("Embedding", "Embedding", np.asarray)
        pred = _get("Clusters_int", "Clusters", _final_labels)

    if emb.size == 0:
        print(f"[MISS] {npz_path}: empty embedding")
        return None

    # Leiden/Louvain labels arrive as strings; factorise to ints for colouring.
    pred = np.asarray(pred).ravel()
    if pred.dtype.kind in "USO":
        pred = np.unique(pred, return_inverse=True)[1]
    pred = pred.astype(int)

    emb = _to_2d_embedding(emb, len(pred), npz_path)

    # Labels_final (when a wrapper writes it) is the post-preprocessing label vector,
    # aligned with Clusters; plain Labels may predate cell filtering and be longer.
    key = "Labels_final" if "Labels_final" in d.files else "Labels"
    y = np.asarray(d[key]).ravel().astype(int) if key in d.files else None
    return np.asarray(emb, dtype=float), pred, y


FIG3C_FILES = {"scVI": "scvi.npz", "scGNN": "scgnn.npz",
               "ADClust": "adclust.npz", "scAce": "scace.npz"}


def load_fig3c_pred(fig3c_dir, label):
    """Cluster assignment for one method from figure 3c's per-method archive.

    scAce stores a nested per-merge-stage history, so 3c takes ``Clusters[-1][-1]``;
    every other method stores a flat vector. Returns None when unavailable.
    """
    fname = FIG3C_FILES.get(label)
    if not fname:
        return None
    path = os.path.join(fig3c_dir, fname)
    if not os.path.isfile(path):
        print(f"[MISS] fig3c assignment for {label}: {path}")
        return None
    d = np.load(path, allow_pickle=True)
    if "Clusters" not in d.files:
        print(f"[MISS] {path}: no 'Clusters'")
        return None
    c = d["Clusters"]
    pred = np.asarray(c[-1][-1]) if label == "scAce" else np.asarray(c)
    pred = pred.ravel()
    if pred.dtype.kind in "USO":
        pred = np.unique(pred, return_inverse=True)[1]
    return pred.astype(int)


def load_umap_cache(path):
    """Return {method: coords} from the manuscript's cached UMAP archive, or {}."""
    if not path:
        return {}
    if not os.path.isfile(path):
        print(f"[WARN] --umap_cache not found: {path}; projecting each run instead")
        return {}
    cache = np.load(path, allow_pickle=True)["UMAP"].item()
    print(f"[INFO] umap cache: " + ", ".join(f"{k}({np.asarray(v).shape[0]})"
                                             for k, v in cache.items()))
    return {k: np.asarray(v) for k, v in cache.items()}


def load_dmvae(dmvae_dir, val, labels_path=None):
    """Return (embedding, predictions, truth) for one DMVAE column.

    ``val`` is either an integer k -- read from ``assignments_all_k.json``, which stores
    one assignment per k over the SAME latent space -- or ``"selected"``, which uses the
    ``Clusters`` array in ``dmvae.npz`` (the model's own choice, as shown in figure 4c).
    """
    npz = os.path.join(dmvae_dir, "dmvae.npz")
    if not os.path.isfile(npz):
        print(f"[MISS] {npz}")
        return None
    d = np.load(npz, allow_pickle=True)
    emb = np.asarray(d["Embedding"], dtype=float)

    if str(val).lower() == "selected":
        pred = np.asarray(d["Clusters"]).ravel().astype(int)
    else:
        js = os.path.join(dmvae_dir, "assignments_all_k.json")
        if not os.path.isfile(js):
            print(f"[MISS] {js}")
            return None
        with open(js) as f:
            assignments = json.load(f)
        if str(val) not in assignments:
            print(f"[MISS] DMVAE k={val}; available: {sorted(map(int, assignments))}")
            return None
        pred = np.asarray(assignments[str(val)], dtype=int)

    y = None
    if labels_path and os.path.isfile(labels_path):
        col = pd.read_csv(labels_path, header=None, sep=r"[,\s]+", engine="python").iloc[:, -1]
        col = col.astype(str).str.strip().str.strip('"')
        y = pd.factorize(col, sort=True)[0]
    elif "Labels" in d.files:
        y = np.asarray(d["Labels"]).ravel().astype(int)

    n = min(len(pred), len(emb), len(y) if y is not None else len(pred))
    if y is not None and len(y) != len(pred):
        print(f"[INFO] DMVAE: truncating to {n} (pred {len(pred)}, labels {len(y)}, emb {len(emb)})")
    return emb[:n], pred[:n], (None if y is None else y[:n])


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def embed_umap(emb, args):
    """Project one run's own latent space. Each run gets its own layout."""
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="euclidean",
        random_state=args.seed,
    )
    return reducer.fit_transform(emb)


def draw(ax, xy, labels, title, args, metrics=None):
    cvec, cmap, uniq, _ = get_colors_cmap(labels)
    ax.scatter(xy[:, 0], xy[:, 1], c=cvec, s=args.point_size, cmap=cmap,
               vmin=-0.5, vmax=len(uniq) - 0.5, linewidths=0)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(0.6)
    if metrics:
        ax.text(0.98, 0.02, metrics, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))


def blank(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8, color="0.45",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("0.8")


def row_label(ax, method):
    """Method name on the leftmost axis of a row. Name only -- the parameter and its
    value are already in each panel title."""
    ax.set_ylabel(method, fontsize=11, fontweight="bold", labelpad=8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    show_truth = not args.no_truth

    # Each row carries its own list of settings, so the grid is as wide as the widest row.
    rows = [
        (label, sub, pname, default, prefix, list(getattr(args, dest)))
        for label, sub, pname, default, prefix, dest in METHOD_SPEC
    ]
    if args.dmvae_dir:
        # sub=None marks the DMVAE row: one embedding, recoloured per k.
        rows.append(("DMVAE", None, "k", "selected", "", list(args.dmvae_k)))
    n_settings = max(len(vals) for _, _, _, _, _, vals in rows)

    # Layout: n_settings prediction columns, then a narrow SPACER column, then a single
    # truth column. The spacer is a real axes kept invisible -- it just opens a gap so the
    # truth column reads as separate from the predictions rather than as a fourth setting.
    SPACER = 0.30                       # spacer width, relative to one panel
    spacer_col = n_settings
    truth_col = n_settings + 1
    ncols = n_settings + (2 if show_truth else 0)
    width_ratios = [1.0] * n_settings + ([SPACER, 1.0] if show_truth else [])

    fig, axes = plt.subplots(
        len(rows), ncols,
        figsize=(3.0 * (n_settings + (1 + SPACER if show_truth else 0)), 3.1 * len(rows)),
        squeeze=False, gridspec_kw={"width_ratios": width_ratios},
    )

    umap_cache = load_umap_cache(args.umap_cache)

    summary = []
    truth_ks = set()
    for r, (label, sub, pname, default, prefix, values) in enumerate(rows):
        if show_truth:
            axes[r][spacer_col].axis("off")     # the gap itself

        # One cached layout for the whole row, when available: every panel is then the
        # SAME projection recoloured, matching the published panel for this method.
        row_xy = umap_cache.get(label)
        if row_xy is not None:
            scope = ("every column" if sub is None else "the default column + truth")
            print(f"[INFO] {label}: cached UMAP layout ({row_xy.shape[0]} cells) for {scope}")

        # The truth panel needs ONE layout, but every setting has its own embedding.
        # Use the method's OWN DEFAULT run, falling back to the first run that loaded.
        truth_candidates = []

        for j in range(n_settings):
            ax_pred = axes[r][j]

            # Rows with fewer settings than the widest row leave trailing cells empty.
            if j >= len(values):
                blank(ax_pred, "")
                continue

            val = values[j]

            if sub is None:                       # ---- DMVAE row ----
                run = load_dmvae(args.dmvae_dir, val, args.dmvae_labels)
                if run is None:
                    blank(ax_pred, f"{label}\nk = {val}\n(missing)")
                    if j == 0:
                        row_label(ax_pred, label)
                    continue
                emb, pred, y = run
                if label in args.drop_truth:
                    y = None
                k_obs = len(np.unique(pred))
                setting = (f"k = {k_obs} (selected)" if str(val).lower() == "selected"
                           else f"k = {val}")
                if y is not None and len(y) == len(pred):
                    truth_k = args.truth_k or len(np.unique(y))
                    truth_ks.add(truth_k)
                    ari, nmi = ARI(y, pred), NMI(y, pred, average_method="arithmetic")
                    metrics = f"ARI {ari:.3f}\nNMI {nmi:.3f}"
                else:
                    truth_k = args.truth_k or truth_k_fallback
                    ari = nmi = float("nan"); metrics = None
                try:
                    xy = (row_xy[:len(pred)] if row_xy is not None and len(row_xy) >= len(pred)
                          else embed_umap(emb, args))
                except Exception as exc:
                    print(f"[FAIL] {label} {setting}: UMAP failed: {exc}")
                    blank(ax_pred, f"{label}\n{setting}\n(UMAP failed)")
                    if j == 0:
                        row_label(ax_pred, label)
                    continue
                # No requested-k line here: every DMVAE column is the same model
                # at a different resolution, so only the resulting k is
                # informative. The selected column says so.
                dmvae_title = f"predicted, k = {k_obs}"
                if str(val).lower() == "selected":
                    dmvae_title += " (selected)"
                draw(ax_pred, xy, pred, dmvae_title, args, metrics)
                if j == 0:
                    row_label(ax_pred, label)
                if y is not None and len(y) == len(pred):
                    truth_candidates.append((str(val).lower() == "selected", xy, y, truth_k, val))
                summary.append({"method": label, "parameter": "k", "default": "selected",
                                "value": str(val), "is_default": str(val).lower() == "selected",
                                "k_observed": int(k_obs), "k_true": int(truth_k),
                                "ARI": float(ari), "NMI": float(nmi)})
                print(f"[OK] {label:8s} k={str(val):<9} -> k={k_obs} (true {truth_k})  "
                      f"ARI={ari:.4f}  NMI={nmi:.4f}")
                continue

            # "default" is the no-flag sentinel: that run passed neither the parameter nor
            # --tag, so its file is the UNTAGGED results_sim_<rep>.npz that
            # run_all_methods.slurm writes.
            is_sentinel = str(val) == "default"
            tag = "" if is_sentinel else f"{prefix}{val}"
            # A panel is "the default" either via the no-flag sentinel or by naming the
            # library default explicitly (scGNN's 0.5 == its 'auto'). Both are marked the
            # same way so one row's middle panel is not labelled differently from another's.
            is_default = is_sentinel or str(val) == str(default)
            shown = default if is_sentinel else val
            setting = f"{pname} = {shown}" + (" (default)" if is_default else "")
            fname = f"results_sim_{args.rep}.npz" if is_sentinel else f"results_sim_{args.rep}__{tag}.npz"
            npz = os.path.join(args.res_root, sub, fname)

            # One bad run must not cost the whole figure -- draw the error and carry on.
            try:
                run = load_run(npz, sub)
            except Exception as exc:
                print(f"[FAIL] {label} {setting}: {type(exc).__name__}: {exc}")
                run = None
                note = f"{label}\n{setting}\n({type(exc).__name__})"
            else:
                note = f"{label}\n{setting}\n(missing)"

            if run is None:
                blank(ax_pred, note)
                if j == 0:
                    row_label(ax_pred, label)
                continue

            emb, pred, y = run

            # Default column: optionally take figure 3c's published assignment instead of
            # this sweep's own default run, so the panel matches the figure exactly.
            src_note = ""
            if is_default and args.fig3c_dir:
                alt = load_fig3c_pred(args.fig3c_dir, label)
                if alt is not None and len(alt) == len(pred):
                    print(f"[INFO] {label}: default column using fig3c assignment "
                          f"(k {len(np.unique(pred))} -> {len(np.unique(alt))})")
                    pred = alt
                    src_note = "fig3c"
                elif alt is not None:
                    print(f"[WARN] {label}: fig3c assignment has {len(alt)} cells but this "
                          f"run has {len(pred)}; keeping the sweep's own default")

            if label in args.drop_truth:
                y = None                      # stored truth is unusable for this method
            k_obs = len(np.unique(pred))

            if y is not None and len(y) == len(pred):
                truth_k = args.truth_k or len(np.unique(y))
                truth_ks.add(truth_k)
                ari, nmi = ARI(y, pred), NMI(y, pred, average_method="arithmetic")
                metrics = f"ARI {ari:.3f}\nNMI {nmi:.3f}"
            else:
                truth_k = args.truth_k or truth_k_fallback
                ari = nmi = float("nan")
                metrics = None

            # Cached layout ONLY for the default column: that is the run the cached
            # coordinates were computed from. Every other value re-trained the model, so
            # its embedding is different and must be projected on its own.
            use_cached = row_xy is not None and is_default and len(row_xy) >= len(pred)
            try:
                xy = row_xy[:len(pred)] if use_cached else embed_umap(emb, args)
            except Exception as exc:
                print(f"[FAIL] {label} {setting}: UMAP failed: {type(exc).__name__}: {exc}")
                blank(ax_pred, f"{label}\n{setting}\n(UMAP failed)")
                if j == 0:
                    row_label(ax_pred, label)
                continue

            title = f"{setting}\npredicted, k = {k_obs}"
            draw(ax_pred, xy, pred, title, args, metrics)
            if j == 0:
                row_label(ax_pred, label)

            if y is not None and len(y) == len(pred):
                truth_candidates.append((is_default, xy, y, truth_k, val))

            summary.append({
                "method": label, "parameter": pname, "default": default, "value": val,
                "is_default": is_default, "source": src_note or "sweep",
                "k_observed": int(k_obs), "k_true": int(truth_k),
                "ARI": float(ari), "NMI": float(nmi),
            })
            print(f"[OK] {label:8s} {pname}={val:<5} -> k={k_obs} (true {truth_k})  "
                  f"ARI={ari:.4f}  NMI={nmi:.4f}")

        # ---- the single truth panel for this row ----
        if show_truth:
            ax_t = axes[r][truth_col]
            chosen = next((c for c in truth_candidates if c[0]), None) or (
                truth_candidates[0] if truth_candidates else None)
            if label in args.drop_truth:
                blank(ax_t, "annotated labels\nnot recoverable\nfor this run")
            elif chosen is None:
                blank(ax_t, "truth\n(no run loaded)")
            else:
                _, xy_t, y_t, truth_k_t, val_t = chosen
                draw(ax_t, xy_t, y_t, f"true labels, k = {truth_k_t}\n"
                                      f"(layout: {pname} = {val_t})", args)

    # True k is read from the data; flag it loudly if runs disagree.
    if len(truth_ks) > 1:
        print(f"[WARN] runs disagree on true k: {sorted(truth_ks)} -- check they share one dataset")

    fig.tight_layout()

    stem = os.path.join(args.out_dir, f"figS7_{args.scenario}_multiresolution_umap")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {stem}.pdf / .png / _summary.json")


if __name__ == "__main__":
    main()
