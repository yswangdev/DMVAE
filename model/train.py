"""Train DMVAE on one dataset with one hyperparameter set.

TrainConfig   -- the hyperparameters run_training needs.
run_training  -- m restarts on one dataset; keeps the lowest-loss run.
save_run      -- write every artifact for a finished run.
"""

import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Import umap (numba/llvmlite) before TensorFlow: TF carries its own LLVM, and with TF
# loaded first llvmlite can bind to those symbols and segfault while numba initialises
# its CPU target.
import umap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Patch
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, TerminateOnNaN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam, RMSprop

from evaluation import finalize_run, save_dmvae_npz, selection_warnings
from model import DMVAE, GetGamma, Sampling, gmmpara_init, p_k_dist
from utils import (
    DMVAETrainingContext,
    add_labels,
    build_autoencoder,
    get_colors_cmap,
    load_pretrain_weights,
    make_dmvae_epoch_callback,
    plot_multi_resolution,
    remap_to_continuous,
)


@dataclass
class TrainConfig:
    """Hyperparameters for one training run.

    The defaults are the RMSprop/MSE pair the grid search selected hyperparameters
    under. ae_optimizer="adam" with ae_loss="binary_crossentropy" and clipnorm=5.
    """

    a: int = 2
    b: int = 15
    truth_k: Optional[int] = None
    epochs: int = 200
    m: int = 1
    ae_lr: float = 1e-4
    ae_epoch: int = 30
    lr_nn: float = 1e-4
    beta: float = 1.0
    batch_size: int = 100
    latent_dim: int = 10
    seed: int = 42
    intermediate_dim: Sequence[int] = field(default_factory=lambda: [500, 500, 2000])
    decay_n: int = 10
    decay_nn: float = 0.9
    alpha: float = 1.0
    ae_optimizer: str = "rmsprop"          # rmsprop | adam
    ae_loss: str = "mean_squared_error"    # mean_squared_error | binary_crossentropy
    clipnorm: Optional[float] = None
    ae_path: str = ""                      # reuse this saved AE instead of training one
    reuse_ae: bool = False
    pretrain: bool = True
    select: str = "map"                    # map | knee
    training_log: bool = False             # per-epoch CSVLogger

    def as_dict(self):
        d = dict(self.__dict__)
        d["intermediate_dim"] = list(self.intermediate_dim)
        return d


def _compile_autoencoder(ae, cfg):
    if cfg.ae_optimizer == "adam":
        opt = Adam(learning_rate=cfg.ae_lr, epsilon=1e-4)
    else:
        opt = RMSprop(learning_rate=cfg.ae_lr, clipnorm=5)
    ae.compile(optimizer=opt, loss=cfg.ae_loss)
    return ae


def _build_dmvae(original_dim, cfg, p_k):
    """DMVAE graph plus the side models the callback and pretraining need."""
    theta_p, u_p, lambda_p = gmmpara_init(cfg.a, cfg.b, cfg.latent_dim)

    x = Input(shape=(original_dim,))
    h = Dense(cfg.intermediate_dim[0], activation="relu")(x)
    h = Dense(cfg.intermediate_dim[1], activation="relu")(h)
    h = Dense(cfg.intermediate_dim[2], activation="relu")(h)
    z_mean = Dense(cfg.latent_dim)(h)
    z_log_var = Dense(cfg.latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    h_dec = Dense(cfg.intermediate_dim[-1], activation="relu")(z)
    h_dec = Dense(cfg.intermediate_dim[-2], activation="relu")(h_dec)
    h_dec = Dense(cfg.intermediate_dim[-3], activation="relu")(h_dec)
    x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_dec)

    p_k_z, p_c_z = GetGamma(theta_p, u_p, lambda_p, cfg.a, cfg.b, cfg.latent_dim)(z)

    gamma_output = Model(inputs=x, outputs=[p_k_z, p_c_z])
    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(inputs=z, outputs=x_decoded_mean, name="decoder")

    dmvae = DMVAE(
        encoder, decoder, theta_p, u_p, lambda_p,
        beta=cfg.beta, original_dim=original_dim, latent_dim=cfg.latent_dim,
        a=cfg.a, b=cfg.b, alpha=cfg.alpha, p_k=p_k,
    )
    return dmvae, gamma_output


def _load_or_fit_autoencoder(X, cfg, out_dir):
    """Return the autoencoder, its latent representation, and its loss trace."""
    ae_dir = os.path.join(out_dir, "ae_sim")
    if cfg.ae_path:
        if not os.path.exists(cfg.ae_path):
            raise FileNotFoundError(f"saved autoencoder not found: {cfg.ae_path}")
        ae = load_model(cfg.ae_path)
        loss_ae = []
    elif cfg.reuse_ae:
        if not os.path.exists(ae_dir):
            raise FileNotFoundError(
                f"--reuse-ae was requested but no saved autoencoder exists at {ae_dir}"
            )
        ae = load_model(ae_dir)
        loss_ae = []
    else:
        ae, _ = build_autoencoder(
            X.shape[1], cfg.intermediate_dim, cfg.latent_dim
        )
        ae = _compile_autoencoder(ae, cfg)
        fitting_ae = ae.fit(
            X, X, epochs=cfg.ae_epoch, batch_size=cfg.batch_size,
            shuffle=True, validation_data=(X, X), verbose=0,
        )
        loss_ae = list(fitting_ae.history["loss"])
        ae.save(ae_dir)

    # The fourth Dense layer is the autoencoder latent layer. Rebuild this view
    # after loading so it always shares the actual trained weights.
    if len(ae.layers) < 5:
        raise ValueError("saved autoencoder does not match the expected DMVAE architecture")
    encoder_ae = Model(ae.input, ae.layers[4].output, name="ae_encoder")
    ae_zmean = encoder_ae.predict(X, batch_size=cfg.batch_size, verbose=0)
    return ae, np.asarray(ae_zmean), loss_ae


def run_training(X, Y, cfg, out_dir, logfile=None):
    """Train ``cfg.m`` restarts on one dataset; keep the lowest-loss one.

    Returns a results dict for :func:`save_run`, or None if every restart hit a
    degenerate p(k).
    """
    if cfg.truth_k is not None and not (cfg.a <= cfg.truth_k <= cfg.b):
        # truth_k indexes the k axis as (truth_k - a); outside [a, b] that silently
        # wraps to the wrong slice, so reject it rather than emit wrong *_t metrics.
        raise ValueError(
            f"truth_k {cfg.truth_k} is outside the k-search range [{cfg.a}, {cfg.b}]"
        )
    if X.ndim != 2 or len(X) != len(Y):
        raise ValueError("X must be a 2-D matrix with one reference label per row")
    if len(X) < cfg.b:
        raise ValueError("Gaussian-mixture initialization requires at least b samples")

    os.makedirs(out_dir, exist_ok=True)
    original_dim = X.shape[1]
    p_k = p_k_dist("uniform", cfg.a, cfg.b)
    start_time = time.time()

    best = {"loss": float("inf"), "run_idx": -1}
    runs = {k: [] for k in ("loss", "acc", "acc_t", "k", "ari", "nmi", "ari_t", "nmi_t", "pk")}

    for j in range(cfg.m):
        print(f"  restart {j}...")
        K.clear_session()
        gc.collect()
        run_seed = cfg.seed + j
        random.seed(run_seed)
        np.random.seed(run_seed)
        tf.keras.utils.set_random_seed(run_seed)

        ae, ae_zmean, loss_ae = _load_or_fit_autoencoder(X, cfg, out_dir)

        dmvae, gamma_output = _build_dmvae(original_dim, cfg, p_k)

        if cfg.pretrain:
            dmvae = load_pretrain_weights(
                dmvae, ae, X, cfg.a, cfg.b, batch_size=cfg.batch_size
            )

        opt_kwargs = {"clipnorm": cfg.clipnorm} if cfg.clipnorm else {}
        rmsprop_nn = RMSprop(learning_rate=cfg.lr_nn, **opt_kwargs)
        dmvae.compile(optimizer=rmsprop_nn)

        trace = {k: [] for k in ("k_list", "k_order_list", "accuracy", "accuracy_t",
                                 "assign", "posteriorK", "ari", "nmi", "ari_t", "nmi_t")}
        dicts = {k: {} for k in ("assign_all", "acc_all", "ari_all", "nmi_all")}
        ctx = DMVAETrainingContext(
            X=X, Y=Y, batch_size=cfg.batch_size, gamma_output=gamma_output, dmvae=dmvae,
            rmsprop_nn=rmsprop_nn, truth_k=cfg.truth_k, a=cfg.a, b=cfg.b,
            latent_dim=cfg.latent_dim, decay_n=cfg.decay_n, decay_nn=cfg.decay_nn,
            logfile=logfile,
            k_list=trace["k_list"], k_order_list=trace["k_order_list"],
            accuracy=trace["accuracy"], accuracy_t=trace["accuracy_t"],
            assign=trace["assign"], posteriorK=trace["posteriorK"],
            assign_all=dicts["assign_all"], acc_all=dicts["acc_all"],
            ari=trace["ari"], nmi=trace["nmi"], ari_t=trace["ari_t"], nmi_t=trace["nmi_t"],
            ari_all=dicts["ari_all"], nmi_all=dicts["nmi_all"],
        )
        epoch_cb = make_dmvae_epoch_callback(ctx)
        callbacks = [epoch_cb, TerminateOnNaN()]
        if cfg.training_log:
            callbacks.insert(1, CSVLogger(os.path.join(out_dir, "training_log.csv"),
                                          append=True))

        fitting = dmvae.fit(
            X, shuffle=True, epochs=cfg.epochs, batch_size=cfg.batch_size,
            verbose=0, callbacks=callbacks,
        )
        if getattr(epoch_cb, "skip", False):
            print(f"  restart {j} skipped: degenerate p(k).")
            continue

        # Reported results describe the model now that fit() has returned; the callback
        # traces stop one optimizer step short of this.
        z_mean, _, _ = dmvae.encoder.predict(X, batch_size=cfg.batch_size, verbose=0)
        p_k_z_final, p_c_z_final = gamma_output.predict(X, batch_size=cfg.batch_size, verbose=0)
        final = finalize_run(p_k_z_final, p_c_z_final, Y, cfg.a, cfg.b,
                             cfg.truth_k, select=cfg.select)

        last_loss = fitting.history["loss"][-1]
        runs["loss"].append({
            "loss": float(last_loss),
            "reconstruction_loss": float(fitting.history["reconstruction_loss"][-1]),
            "kl_loss": float(fitting.history["kl_loss"][-1]),
        })
        for key in ("acc", "acc_t", "ari", "nmi", "ari_t", "nmi_t", "pk"):
            runs[key].append(final[key])
        runs["k"].append(final["k_index"])

        if last_loss < best["loss"]:
            reducer = umap.UMAP(
                n_neighbors=min(15, len(X) - 1), min_dist=0.1,
                metric="euclidean", random_state=cfg.seed, n_jobs=1,
            )
            best = {
                "loss": last_loss,
                "run_idx": j,
                "final": final,
                "z_mean": z_mean.copy(),
                "ae_zmean": np.asarray(ae_zmean).copy(),
                "loss_curve": list(fitting.history["loss"]),
                "recon_loss": list(fitting.history["reconstruction_loss"]),
                "kl_loss": list(fitting.history["kl_loss"]),
                "loss_ae": loss_ae,
                # Per-epoch history.
                "acc": list(trace["accuracy"]), "acc_t": list(trace["accuracy_t"]),
                "k": list(trace["k_list"]),
                "ari": list(trace["ari"]), "nmi": list(trace["nmi"]),
                "ari_t": list(trace["ari_t"]), "nmi_t": list(trace["nmi_t"]),
                # Converged results.
                "p_c_z": p_c_z_final, "p_k_z": p_k_z_final,
                "embedding": reducer.fit_transform(z_mean),
                "ae_embedding": reducer.fit_transform(ae_zmean),
            }
        print(f"  restart {j} done.")

    if best["run_idx"] < 0:
        print("  every restart was skipped (degenerate p(k)); nothing saved.")
        return None

    best["runs"] = runs
    best["time_use"] = time.time() - start_time
    best["cfg"] = cfg
    return best


def _save_posterior_meta(out_dir, res, cfg):
    """Per-k padding diagnostic for the saved posterior.

    GetGamma floors the unnormalised density at 1e-10 before normalising, so when every
    component underflows that floor the posterior collapses to uniform 1/b. The tell is
    the padded components holding (b - k)/b of the mass, so record how far each k-slice
    sits from that degenerate case before reading anything into a graded posterior.
    """
    p_c_z, a, b = res["p_c_z"], cfg.a, cfg.b
    pad = {}
    for i, k_value in enumerate(range(a, b + 1)):
        mass = p_c_z[:, i, k_value:].sum(axis=1) if k_value < b else np.zeros(p_c_z.shape[0])
        degenerate = (b - k_value) / b
        pad[str(k_value)] = {
            "padding_mass_mean": float(mass.mean()),
            "padding_mass_max": float(mass.max()),
            "degenerate_value": float(degenerate),
            "frac_cells_near_degenerate": float(np.mean(np.abs(mass - degenerate) < 1e-3)),
        }
    with open(os.path.join(out_dir, "p_c_z_meta.json"), "w") as f:
        json.dump({
            "shape": list(p_c_z.shape),
            "axes": ["cell", "k_hypothesis", "component"],
            "a": a, "b": b, "k_values": list(range(a, b + 1)), "k_index_offset": a,
            "truth_k": cfg.truth_k,
            "truth_k_index": (cfg.truth_k - a) if cfg.truth_k is not None else None,
            "k_selected": res["final"]["k_selected"],
            "padding_diagnostic": pad,
        }, f, indent=2)


def save_run(out_dir, X, Y, res, *, multi_resolution=False, legacy_artifacts=False):
    """Write the run.

    By default this is one file, ``dmvae.npz``. ``multi_resolution`` adds the
    assignment at every candidate k and renders one UMAP per k; ``legacy_artifacts``
    also writes the loose per-quantity files the manuscript figure scripts read.
    """
    cfg, final, runs = res["cfg"], res["final"], res["runs"]
    os.makedirs(out_dir, exist_ok=True)

    def p(name):
        return os.path.join(out_dir, name)

    for msg in selection_warnings(final["selection"]):
        print(f"[select_k] {msg}")

    save_dmvae_npz(
        p("dmvae.npz"),
        embedding=res["z_mean"], final=final, a=cfg.a, b=cfg.b, labels_true=Y,
        umap_2d=res["embedding"], multi_resolution=multi_resolution,
        traces={"ari": res["ari"], "nmi": res["nmi"], "k": res["k"], "acc": res["acc"]},
        best_loss=res["loss"], best_run_idx=res["run_idx"], time_use=res["time_use"],
    )

    if multi_resolution:
        plot_multi_resolution(out_dir, np.asarray(res["embedding"])[:, :2],
                              final["assign_all"])

    if not legacy_artifacts:
        return

    with open(p("hyperparameters.json"), "w") as f:
        json.dump(cfg.as_dict(), f, indent=4)

    # Per-epoch history of the selected run.
    np.savetxt(p("z_mean.txt"), res["z_mean"])
    np.savetxt(p("DMVAE_loss.txt"), res["loss_curve"])
    np.savetxt(p("accuracy.txt"), res["acc"])
    np.savetxt(p("accuracy_t.txt"), res["acc_t"])
    np.savetxt(p("k.txt"), res["k"])
    np.savetxt(p("ari.txt"), res["ari"])
    np.savetxt(p("nmi.txt"), res["nmi"])
    np.savetxt(p("ari_t.txt"), res["ari_t"])
    np.savetxt(p("nmi_t.txt"), res["nmi_t"])

    # Converged results.
    np.savetxt(p("assign_c.txt"), final["assign"])
    np.savetxt(p("labels_true.txt"), Y, fmt="%d")
    np.savetxt(p("best_ari_last.txt"), np.atleast_1d(final["ari"]))
    np.savetxt(p("best_nmi_last.txt"), np.atleast_1d(final["nmi"]))

    pk = np.asarray(final["pk"])
    np.savetxt(p("posteriorK_best.txt"), np.atleast_2d(pk))
    np.savetxt(p("posteriorK_best_normalized.txt"), np.atleast_2d(pk / max(len(Y), 1)))
    np.save(p("posteriorK_all_iters.npy"), np.stack(runs["pk"]))

    np.save(p("p_c_z_best.npy"), res["p_c_z"])
    np.save(p("p_k_z_best.npy"), res["p_k_z"])
    np.save(p("dmvae_umap_2d.npy"), np.asarray(res["embedding"]))
    np.save(p("ae_latent.npy"), res["ae_zmean"])
    np.save(p("ae_umap_2d.npy"), np.asarray(res["ae_embedding"]))
    _save_posterior_meta(out_dir, res, cfg)

    # Per-k breakdown, all from the one post-fit posterior.
    with open(p("assignments_all_k.json"), "w") as f:
        json.dump({str(k): np.asarray(v).astype(int).tolist()
                   for k, v in final["assign_all"].items()}, f, indent=2)
    with open(p("accuracies_all_k.json"), "w") as f:
        json.dump({str(k): float(v) for k, v in final["acc_all"].items()}, f, indent=2)
    with open(p("ari_all.json"), "w") as f:
        json.dump({int(k): float(v) for k, v in final["ari_all"].items()}, f, indent=4)
    with open(p("nmi_all.json"), "w") as f:
        json.dump({int(k): float(v) for k, v in final["nmi_all"].items()}, f, indent=4)

    # One entry per restart.
    with open(p("all_loss.json"), "w") as f:
        json.dump(runs["loss"], f, indent=4)
    for name, key in [("all_accuracy", "acc"), ("all_accuracy_t", "acc_t"), ("all_k", "k"),
                      ("all_ari", "ari"), ("all_nmi", "nmi"),
                      ("all_ari_t", "ari_t"), ("all_nmi_t", "nmi_t")]:
        np.savetxt(p(f"{name}.txt"), np.array(runs[key]))

    adjusted_assign, adjusted_k = remap_to_continuous(final["assign"])
    _save_plots(out_dir, Y, res, adjusted_assign, adjusted_k)


def _umap_panel(path, xy, pred, Y, title, annotation=None):
    """Two-panel UMAP: predicted labels on top, true labels below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    for ax, labels, name in ((ax1, pred, title), (ax2, Y, "True Label")):
        cvec, cmap, uniq, cols = get_colors_cmap(labels)
        ax.scatter(xy[:, 0], xy[:, 1], c=cvec, s=1, cmap=cmap, vmin=-0.5, vmax=len(uniq) - 0.5)
        add_labels(ax, xy, labels, {u: cols[i] for i, u in enumerate(uniq)})
        ax.set_title(name, fontsize=24)
        ax.legend(handles=[Patch(facecolor=cols[i], edgecolor="gray", label=str(u))
                           for i, u in enumerate(uniq)],
                  loc="center left", bbox_to_anchor=(1, 0.5), title="Classes", fontsize=8)
    if annotation:
        ax1.text(0.98, 0.02, annotation, transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=12, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.clf()


def _save_plots(out_dir, Y, res, adjusted_assign, adjusted_k):
    final = res["final"]
    xy = np.asarray(res["embedding"])[:, :2]

    _umap_panel(
        os.path.join(out_dir, "umap_DMVAE_best.png"), xy, adjusted_assign, Y,
        f"Predicted Label (k={adjusted_k})",
        f"ACC: {final['acc']:.3f}\nARI: {final['ari']:.3f}\nNMI: {final['nmi']:.3f}",
    )
    for k_value, assign_k in final["assign_all"].items():
        _umap_panel(
            os.path.join(out_dir, f"umap_DMVAE{k_value}.png"), xy, np.asarray(assign_k), Y,
            f"Predicted Label (k={k_value})",
            f"ACC: {final['acc_all'][k_value]:.3f}\nARI: {final['ari_all'][k_value]:.3f}\n"
            f"NMI: {final['nmi_all'][k_value]:.3f}",
        )

    ae_xy = np.asarray(res["ae_embedding"])
    plt.figure(figsize=(8, 6))
    plt.scatter(ae_xy[:, 0], ae_xy[:, 1], c=Y, s=1)
    plt.colorbar()
    plt.title("AE Latent Representation")
    plt.savefig(os.path.join(out_dir, "ae.png"))
    plt.close()
    plt.clf()

    for curves, labels, title, fname in (
        ([res["loss_ae"]], ["AE loss"], "AE Loss", "ae_loss.png"),
        ([res["loss_curve"], res["recon_loss"], res["kl_loss"]],
         ["Total Loss", "Reconstruction Loss", "KL Loss"], "DMVAE Model Loss", "DMVAE_loss.png"),
    ):
        if not any(len(c) for c in curves):
            continue
        plt.figure(figsize=(8, 6))
        for c, lab in zip(curves, labels):
            plt.plot(c, label=lab)
        plt.legend()
        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
        plt.clf()
