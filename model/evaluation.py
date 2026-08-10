"""Clustering metrics, cluster-number selection, and the dmvae.npz archive.

cluster_acc    -- Hungarian-matched clustering accuracy.
select_k       -- read the cluster number off S(k), by maximum or by knee.
finalize_run   -- converged results from the post-fit posterior.
save_dmvae_npz -- write the run archive.

A run writes one file, dmvae.npz::

    d = np.load("dmvae.npz")
    d["embedding"]    # (N, D)   latent means
    d["clusters"]     # (N,)     assignment at the selected k, labels 0..K-1
    d["S_k"]          # (b-a+1,) aggregate support, aligned to d["k_values"]
    d["k_selected"]

No TensorFlow here, so figure and analysis code can import it cheaply.
"""

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as _ARI
from sklearn.metrics import normalized_mutual_info_score as _NMI
import numpy as np


# --- Metrics ---


def cluster_acc(Y_pred, Y):
    """Hungarian-matched accuracy for arbitrary finite integer label values."""
    Y_pred = np.asarray(Y_pred).ravel()
    Y = np.asarray(Y).ravel()
    if Y_pred.size != Y.size:
        raise ValueError("predicted and reference labels must have the same length")
    if Y_pred.size == 0:
        raise ValueError("clustering accuracy is undefined for empty label arrays")
    _, pred = np.unique(Y_pred, return_inverse=True)
    _, truth = np.unique(Y, return_inverse=True)
    D = max(pred.max(), truth.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    np.add.at(w, (pred, truth), 1)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    matched = w[row_ind, col_ind].sum()
    return float(matched / Y_pred.size), list(zip(row_ind, col_ind))


def remap_to_continuous(labels):
    """Remap arbitrary integer cluster labels to a contiguous 0..K-1 range.

    Empty clusters (labels in ``[0, max(labels)]`` with zero samples) are dropped,
    so the returned assignment has no gaps. Order is preserved by the original
    label value via ``np.unique``.

    Returns
    -------
    adjusted : np.ndarray
        Same shape as ``labels``, with values in ``[0, n_unique - 1]``.
    n_clusters : int
        Number of clusters that actually contain samples.
    """
    labels_arr = np.asarray(labels)
    uniq = np.unique(labels_arr)
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    adjusted = np.array([label_to_idx[l] for l in labels_arr.ravel()], dtype=np.int64)
    adjusted = adjusted.reshape(labels_arr.shape)
    return adjusted, int(len(uniq))


# --- Selecting the cluster number ---


#: Below this the normalised S(k) curve is near-linear and the knee is not identified.
WEAK_KNEE = 0.10


def _normalise(ks, s):
    """Map both axes onto [0, 1]."""
    x = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    y = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    return x, y


def knee_of(s_k, a):
    """Knee of S(k): the point furthest from the chord joining the endpoints.

    Returns ``(k, distance, strength)`` -- the perpendicular offset from the chord
    at the selected k, and the Kneedle ``max(y - x)`` magnitude.
    """
    s = np.asarray(s_k, dtype=float).ravel()
    ks = np.arange(a, a + len(s))
    if len(s) < 3:
        return int(ks[int(np.argmax(s))]), 0.0, 0.0

    x, y = _normalise(ks, s)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom < 1e-12:
        return int(ks[0]), 0.0, 0.0
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    i = int(np.argmax(dist))
    return int(ks[i]), float(dist[i]), float(np.max(y - x))


def select_k(s_k, a, rule="map"):
    """Pick the cluster number from S(k) under ``rule`` ("map" or "knee")."""
    if rule not in ("map", "knee"):
        raise ValueError(f"rule must be 'map' or 'knee', got {rule!r}")

    s = np.asarray(s_k, dtype=float).ravel()
    ks = np.arange(a, a + len(s))
    k_map = int(ks[int(np.argmax(s))])
    k_knee, knee_distance, knee_strength = knee_of(s, a)

    return {
        "k_selected": k_map if rule == "map" else k_knee,
        "selection_rule": rule,
        "k_map": k_map,
        "k_knee": k_knee,
        "knee_distance": knee_distance,
        "knee_strength": knee_strength,
        # A maximum at a or b means the range should be widened.
        "map_at_boundary": bool(k_map in (int(ks[0]), int(ks[-1]))),
        "weak_knee": bool(knee_strength < WEAK_KNEE),
    }


def selection_warnings(selection):
    """Caveats worth printing at the end of a run."""
    msgs = []
    if selection["selection_rule"] == "map" and selection["map_at_boundary"]:
        msgs.append(
            f"argmax S(k) sits at the edge of the candidate range (k={selection['k_map']}). "
            "S(k) never turned over, so widen [a, b] and refit, or use --select knee.")
    if selection["selection_rule"] == "knee" and selection["weak_knee"]:
        msgs.append(
            f"knee strength {selection['knee_strength']:.3f} < {WEAK_KNEE}: the normalised "
            "S(k) curve is near-linear, so this knee is not well identified.")
    if selection["k_map"] != selection["k_knee"]:
        msgs.append(
            f"argmax S(k) = {selection['k_map']} but the knee is at k = {selection['k_knee']}; "
            "inspect the shape of S(k) before reporting either.")
    return msgs


# --- Run results ---


def _assign_for_hypothesis(p_c_z, cl, a):
    """Hard assignment under k-hypothesis index ``cl``, ignoring padding components."""
    n_real = a + cl
    return np.argmax(np.asarray(p_c_z)[:, cl, :n_real], axis=1)


def finalize_run(p_k_z, p_c_z, Y, a, b, truth_k=None, select="map"):
    """Converged results from the post-fit posterior.

    Parameters
    ----------
    p_k_z : array (n_cells, b - a + 1)
        Post-fit posterior over k hypotheses.
    p_c_z : array (n_cells, b - a + 1, b)
        Post-fit posterior over components, per k hypothesis.
    Y : array (n_cells,)
        Ground-truth labels.
    a, b : int
        Bounds of the k search.
    truth_k : int, optional
        Ground-truth k. Truth-k metrics are NaN when it falls outside ``[a, b]``.
    select : {"map", "knee"}
        Rule for reading k off S(k). Both candidates are recorded either way.

    Returns
    -------
    dict
        ``k_index``     selected hypothesis index
        ``k_selected``  ``a + k_index``, the hypothesis the rule picked
        ``s_k``         aggregate support S(k) = pk / n_cells
        ``selection``   the full dict from :func:`select_k`
        ``k_distinct``  number of distinct clusters in ``assign``
        ``assign``, ``acc``, ``ari``, ``nmi``   at the selected k
        ``assign_all``, ``acc_all``, ``ari_all``, ``nmi_all``   keyed by actual k
        ``assign_truth``, ``acc_t``, ``ari_t``, ``nmi_t``   at ``truth_k``
        ``pk``   summed responsibility per k hypothesis
    """
    p_k_z = np.asarray(p_k_z)
    p_c_z = np.asarray(p_c_z)
    Y = np.asarray(Y)

    pk = p_k_z.sum(axis=0)
    # pk is the summed responsibility per k, so it totals n_cells; dividing gives
    # S(k) = (1/N) sum_i p(k | z_i).
    s_k = pk / max(len(Y), 1)
    selection = select_k(s_k, a, rule=select)
    k_index = int(selection["k_selected"]) - a

    assign = _assign_for_hypothesis(p_c_z, k_index, a)
    out = {
        "k_index": k_index,
        "k_selected": a + k_index,
        "s_k": s_k,
        "selection": selection,
        "k_distinct": int(np.unique(assign).size),
        "assign": assign,
        "acc": float(cluster_acc(assign, Y)[0]),
        "ari": float(_ARI(Y, assign)),
        "nmi": float(_NMI(Y, assign, average_method="arithmetic")),
        "pk": pk,
    }

    assign_all, acc_all, ari_all, nmi_all = {}, {}, {}, {}
    for cl in range(0, b - a + 1):
        k_value = a + cl
        assign_k = _assign_for_hypothesis(p_c_z, cl, a)
        assign_all[k_value] = assign_k
        acc_all[k_value] = float(cluster_acc(assign_k, Y)[0])
        ari_all[k_value] = float(_ARI(Y, assign_k))
        nmi_all[k_value] = float(_NMI(Y, assign_k, average_method="arithmetic"))
    out.update(assign_all=assign_all, acc_all=acc_all, ari_all=ari_all, nmi_all=nmi_all)

    if truth_k is not None and a <= truth_k <= b:
        assign_t = assign_all[truth_k]
        out.update(
            assign_truth=assign_t,
            acc_t=acc_all[truth_k],
            ari_t=ari_all[truth_k],
            nmi_t=nmi_all[truth_k],
        )
    else:
        out.update(assign_truth=None, acc_t=np.nan, ari_t=np.nan, nmi_t=np.nan)

    return out


def save_dmvae_npz(path, *, embedding, final, a, b, labels_true=None,
                   umap_2d=None, multi_resolution=False, traces=None,
                   best_loss=None, best_run_idx=None, time_use=None):
    """Write ``dmvae.npz``, the one file a finished run produces.

    ``final`` is the dict from :func:`finalize_run`. Only the selected k is stored
    unless ``multi_resolution``, which also stores the per-k assignment stack and the
    2-D UMAP those are drawn on.

    ``traces`` holds the per-epoch history (ARI/NMI/K/ACC). It differs from the
    converged values by one optimizer step, so reduce with ``BestARI`` and never with
    ``mean()`` over a trace, which averages in the unconverged early epochs.
    """
    selection = final["selection"]
    k_selected = int(selection["k_selected"])
    clusters, n_clusters = remap_to_continuous(final["assign_all"][k_selected])
    k_values = np.arange(a, b + 1, dtype=np.int64)

    payload = {
        "embedding": np.asarray(embedding, dtype=np.float32),
        "clusters": clusters.astype(np.int64),
        "S_k": np.asarray(final["s_k"], dtype=float),
        "k_values": k_values,
        "k_selected": np.array(k_selected),
        "n_clusters": np.array(n_clusters),
        "a": np.array(int(a)),
        "b": np.array(int(b)),
        "selection_rule": np.array(str(selection["selection_rule"])),
        "k_map": np.array(int(selection["k_map"])),
        "k_knee": np.array(int(selection["k_knee"])),
        "knee_strength": np.array(float(selection["knee_strength"])),
        "map_at_boundary": np.array(bool(selection["map_at_boundary"])),
    }

    if multi_resolution:
        # (N, b - a + 1): column j is the assignment at k = a + j, each remapped
        # independently so labels run 0..K_j-1 within a column.
        stack = np.empty((len(clusters), len(k_values)), dtype=np.int64)
        for j, k in enumerate(k_values):
            stack[:, j], _ = remap_to_continuous(final["assign_all"][int(k)])
        payload["clusters_all_k"] = stack
        if umap_2d is not None:
            payload["umap_2d"] = np.asarray(umap_2d, dtype=np.float32)

    # Present only when the caller had reference labels.
    if labels_true is not None:
        payload["labels_true"] = np.asarray(labels_true).astype(np.int64).ravel()
        payload["BestARI"] = np.array(final["ari"])
        payload["BestNMI"] = np.array(final["nmi"])
        payload["BestACC"] = np.array(final["acc"])
    for name, key in (("ARI", "ari"), ("NMI", "nmi"), ("K", "k"), ("ACC", "acc")):
        if traces and key in traces:
            payload[name] = np.array(traces[key])

    for name, value in (("BestLoss", best_loss), ("BestRunIndex", best_run_idx),
                        ("Time_use", time_use)):
        if value is not None:
            payload[name] = np.array(value)

    np.savez(path, **payload)
    return path
