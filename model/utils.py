"""Shared training helpers.

build_autoencoder, init_gmm_priors, load_pretrain_weights -- SAE pretraining.
DMVAETrainingContext, make_dmvae_epoch_callback, bad_pk   -- per-epoch callback.
get_colors_cmap, add_labels, plot_multi_resolution        -- plotting.
"""

from dataclasses import dataclass
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.mixture import GaussianMixture as GMM
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from typing import Any, Dict, List, Optional, TextIO
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from evaluation import cluster_acc, remap_to_continuous  # noqa: F401  (re-exported)


# --- Autoencoder pretraining and GMM prior init ---


def build_autoencoder(original_dim, intermediate_dim, latent_dim):
    """Symmetric dense autoencoder. Returns ``(autoencoder, latent_encoder)``.

    The layer order matters: :func:`load_pretrain_weights` copies encoder layers 1-4
    and decoder layers -1..-4 by position.
    """
    x_in = Input(shape=(original_dim,))
    h = Dense(intermediate_dim[0], activation="relu")(x_in)
    h = Dense(intermediate_dim[1], activation="relu")(h)
    h = Dense(intermediate_dim[2], activation="relu")(h)
    latent = Dense(latent_dim, activation="relu")(h)
    h_dec = Dense(intermediate_dim[-1], activation="relu")(latent)
    h_dec = Dense(intermediate_dim[-2], activation="relu")(h_dec)
    h_dec = Dense(intermediate_dim[-3], activation="relu")(h_dec)
    x_out = Dense(original_dim, activation="sigmoid")(h_dec)
    return Model(x_in, x_out), Model(x_in, latent, name="encoder")


def init_gmm_priors(sample, u_p, lambda_p, a, b):
    """Fit one diagonal GMM per k in ``[a, b]`` and write means/variances into the priors.

    Component slots beyond ``k`` are padding: means are set to 1e-10 and variances to 1
    so the padded components carry no mass.
    """
    for n_centroid in range(a, b + 1):
        g = GMM(
            n_components=n_centroid,
            covariance_type="diag",
            random_state=42,
            reg_covar=1e-3,
            n_init=5,
        )
        g.fit(sample)

        means = tf.transpose(tf.convert_to_tensor(g.means_, dtype=tf.float32))
        means_padded = tf.pad(means, [[0, 0], [0, b - n_centroid]], constant_values=1e-10)
        u_p[n_centroid - a, :, :].assign(means_padded)

        covs = tf.transpose(tf.convert_to_tensor(g.covariances_, dtype=tf.float32))
        covs_padded = tf.pad(covs, [[0, 0], [0, b - n_centroid]], constant_values=1)
        lambda_p[n_centroid - a, :, :].assign(covs_padded)


def load_pretrain_weights(dmvae, ae, X, a, b, batch_size=None):
    """Copy autoencoder weights into ``dmvae`` and initialise its GMM priors.

    The per-k GMMs are fitted to the DMVAE encoder mean *after* copying the trained
    weights. This matters because the autoencoder latent layer uses ReLU while the
    DMVAE mean layer is linear.
    """
    for i in (1, 2, 3, 4):
        dmvae.encoder.layers[i].set_weights(ae.layers[i].get_weights())
    for i in (-1, -2, -3, -4):
        dmvae.decoder.layers[i].set_weights(ae.layers[i].get_weights())

    latent_sample = dmvae.encoder.predict(
        X, batch_size=batch_size, verbose=0
    )[0]
    init_gmm_priors(np.asarray(latent_sample), dmvae.u_p, dmvae.lambda_p, a, b)
    print("Pretrain weights loaded!")
    return dmvae


# --- Per-epoch training callback ---


@dataclass
class DMVAETrainingContext:
    X: np.ndarray
    Y: np.ndarray
    batch_size: int
    gamma_output: Any
    dmvae: Any
    rmsprop_nn: Any
    truth_k: Optional[int]
    a: int
    b: int
    latent_dim: int
    decay_n: int
    decay_nn: float
    logfile: Optional[TextIO]
    k_list: List
    k_order_list: List
    accuracy: List
    accuracy_t: List
    assign: List
    posteriorK: List
    assign_all: Dict[int, np.ndarray]
    acc_all: Dict[int, float]
    ari: List
    nmi: List
    ari_t: List
    nmi_t: List
    ari_all: Dict[int, float]
    nmi_all: Dict[int, float]


def bad_pk(pk) -> bool:
    """Return True if pk is None/empty/has NaNs/inf (mirrors grid_search)."""
    if pk is None:
        return True
    pk = np.asarray(pk)
    if pk.size == 0:
        return True
    if not np.all(np.isfinite(pk)):
        return True
    return False


def make_dmvae_epoch_callback(ctx: DMVAETrainingContext) -> Callback:
    # Only the nn optimizer trains the model, so it is the only one decayed.
    def lr_decay():
        current_lr_nn = ctx.rmsprop_nn.learning_rate.numpy()
        new_lr_nn = max(current_lr_nn * ctx.decay_nn, 1e-7)
        ctx.rmsprop_nn.learning_rate.assign(new_lr_nn)
        print("lr_nn: %f" % ctx.rmsprop_nn.learning_rate.numpy())

    truth_in_range = (ctx.truth_k is not None) and (ctx.a <= ctx.truth_k <= ctx.b)

    class EpochBegin(Callback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.skip = False

        def on_epoch_begin(self, epoch, logs=None):
            if epoch % ctx.decay_n == 0 and epoch != 0:
                lr_decay()

            gamma = ctx.gamma_output.predict(ctx.X, batch_size=ctx.batch_size, verbose=0)
            p_k_z, p_c_z = gamma[0], gamma[1]

            k = np.argmax(tf.reduce_sum(p_k_z, axis=0), axis=-1)
            pk = tf.reduce_sum(p_k_z, axis=0)
            ctx.k_list.append(k)
            k_order = tf.argsort(tf.reduce_sum(p_k_z, axis=0), direction="DESCENDING")
            ctx.k_order_list.append(k_order.numpy())
            # Argmax over the real components only. Columns from a+k onward are
            # padding held near zero by the epoch-end reset, and must never win.
            p_c_label = p_c_z[:, k, : ctx.a + k]
            assign_c = np.argmax(p_c_label, axis=1)
            ctx.assign.append(assign_c)
            acc = cluster_acc(assign_c, ctx.Y)
            ctx.accuracy.append(acc[0])
            ctx.posteriorK.append(pk.numpy())

            ctx.ari.append(ARI(ctx.Y, assign_c))
            ctx.nmi.append(NMI(ctx.Y, assign_c, average_method="arithmetic"))

            # Truth-k metrics only if truth_k falls inside [a, b]; otherwise pad with
            # NaN so the per-epoch arrays stay aligned with the selected-k ones.
            if truth_in_range:
                p_truth_label = p_c_z[:, ctx.truth_k - ctx.a, : ctx.truth_k]
                assign_truth = np.argmax(p_truth_label, axis=1)
                acc_t = cluster_acc(assign_truth, ctx.Y)
                ctx.accuracy_t.append(acc_t[0])
                ctx.ari_t.append(ARI(ctx.Y, assign_truth))
                ctx.nmi_t.append(NMI(ctx.Y, assign_truth, average_method="arithmetic"))
            else:
                ctx.accuracy_t.append(np.nan)
                ctx.ari_t.append(np.nan)
                ctx.nmi_t.append(np.nan)

            for cl in range(0, ctx.b - ctx.a + 1):
                k_value = cl + ctx.a
                label = p_c_z[:, cl, :k_value]
                ctx.assign_all[k_value] = np.argmax(label, axis=1)
                ctx.acc_all[k_value] = cluster_acc(ctx.assign_all[k_value], ctx.Y)[0]

            for k_value in range(ctx.a, ctx.b + 1):
                ctx.ari_all[k_value] = ARI(ctx.Y, ctx.assign_all[k_value])
                ctx.nmi_all[k_value] = NMI(ctx.Y, ctx.assign_all[k_value], average_method="arithmetic")

            if epoch > 0:
                if ctx.logfile is not None:
                    try:
                        ctx.logfile.write(f"epoch={epoch} | k_order={k_order.numpy().tolist()}\n")
                        ctx.logfile.write(f"epoch={epoch} | acc={acc[0]:0.8f}\n")
                        ctx.logfile.write(f"epoch={epoch} | pk={pk.numpy().tolist()}\n")
                    except OSError:
                        pass
                # A degenerate pk poisons every downstream metric; stop this run so the
                # caller can discard it instead of saving NaN artifacts.
                if bad_pk(pk.numpy()):
                    print("pk is invalid (NaN/Inf/empty). Skipping this iteration.\n")
                    self.skip = True
                    self.model.stop_training = True
                    return

        def on_epoch_end(self, epoch, logs=None):
            for i in range(ctx.b - ctx.a):
                ctx.dmvae.theta_p[i, i + ctx.a : ctx.b].assign(
                    tf.constant(1e-10, shape=[ctx.b - (i + ctx.a)], dtype=tf.float32)
                )
                ctx.dmvae.u_p[i, :, i + ctx.a : ctx.b].assign(
                    tf.constant(1e-10, shape=[ctx.latent_dim, ctx.b - (i + ctx.a)], dtype=tf.float32)
                )
                ctx.dmvae.lambda_p[i, :, i + ctx.a : ctx.b].assign(
                    tf.constant(1, shape=[ctx.latent_dim, ctx.b - (i + ctx.a)], dtype=tf.float32)
                )

    return EpochBegin()


# --- Plotting helpers ---


DISTINCT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
    "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3",
]


def get_colors_cmap(labels):
    uniq = np.unique(labels)
    n = len(uniq)
    colors = DISTINCT_COLORS[:n] if n <= len(DISTINCT_COLORS) else plt.cm.tab20(np.linspace(0, 1, n))
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    cvec = np.array([label_to_idx[l] for l in labels])
    cmap = ListedColormap(colors)
    return cvec, cmap, uniq, colors




def add_labels(ax, xy, labels, colors_by_label, fontsize=9):
    xy, labels = np.asarray(xy), np.asarray(labels)
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() == 0:
            continue
        x, y = np.median(xy[mask], axis=0)
        ax.text(
            x, y, str(lab), fontsize=fontsize, fontweight="bold", ha="center", va="center",
            color="black", path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )


def plot_multi_resolution(out_dir, umap_2d, assignments_by_k, point_size=1.0, dpi=150):
    """One UMAP per candidate k, coloured by that k's assignment.

    The same point cloud throughout; only the colouring changes with k. No ARI, NMI
    or accuracy is drawn -- these plots do not depend on reference labels.
    """
    import os

    xy = np.asarray(umap_2d)
    written = []
    for k in sorted(assignments_by_k):
        pred, _ = remap_to_continuous(np.asarray(assignments_by_k[k]).ravel())
        cvec, cmap, uniq, cols = get_colors_cmap(pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(xy[:, 0], xy[:, 1], c=cvec, s=point_size, cmap=cmap,
                   vmin=-0.5, vmax=len(uniq) - 0.5)
        add_labels(ax, xy, pred, {u: cols[i] for i, u in enumerate(uniq)})
        ax.set_title(f"k = {k}", fontsize=18)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(handles=[Patch(facecolor=cols[i], edgecolor="gray", label=str(u))
                           for i, u in enumerate(uniq)],
                  loc="center left", bbox_to_anchor=(1, 0.5), title="Cluster", fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, f"umap_k{k}.png")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        written.append(path)
    return written
