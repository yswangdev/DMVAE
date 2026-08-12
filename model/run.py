"""Command-line entry point: load data, loop, call run_training.

main        -- parse args, loop over datasets and hyperparameter combinations.
build_parser -- the CLI.

"""

import argparse
import importlib.metadata
import itertools
import os
import re
import sys


def _num_list(text, typ=float):
    """Parse '1e-3,1e-4' or '1e-3 1e-4' into [1e-3, 1e-4]."""
    return [typ(p) for p in re.split(r"[,\s]+", str(text).strip()) if p]


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    d = p.add_argument_group("data")
    d.add_argument("--input-datafile", "--input_datafile", required=True,
                   help="directory holding the data files")
    d.add_argument("--input-file", "--input_file", required=True,
                   help="expression filename; may contain {i} to run a series, "
                        "e.g. simnorm_{i}.txt with --start/--end")
    d.add_argument("--meta-file", "--meta_file", required=True,
                   help="label filename, same {i} convention as --input-file")
    d.add_argument("--start", type=int, default=1, help="first {i}, inclusive")
    d.add_argument("--end", type=int, default=2, help="last {i}, exclusive")

    o = p.add_argument_group("output")
    o.add_argument("--output-base-path", "--output_base_path", "--results-base", "--grid-base",
               dest="output_base_path", required=True,
               help="directory for run results (--results-base/--grid-base are "
                    "accepted aliases; if both are given the last one wins)")
    o.add_argument("--output-hyp", "--output_hyp", default=None,
                   help="directory for the callback log (default: --output-base-path)")
    o.add_argument("--select", choices=["map", "knee"], default="map",
                   help="rule for reading the cluster number off S(k): the maximum "
                        "(map) or the knee. Both are recorded either way")
    o.add_argument("--multi-resolution", dest="multi_resolution", action="store_true",
                   help="also keep the assignment at every candidate k and render "
                        "one UMAP per k")
    o.add_argument("--legacy-artifacts", dest="legacy_artifacts", action="store_true",
                   help="also write the loose files the manuscript figure scripts read")

    h = p.add_argument_group(
        "hyperparameters (give several values to run a grid, e.g. --lr-nn '1e-3,1e-4')")
    h.add_argument("--ae-lr", "--ae_lr", "--ae-lr-grid", dest="ae_lr", default="1e-4")
    h.add_argument("--ae-epoch", "--ae_epoch", "--ae-epoch-grid", dest="ae_epoch", default="30")
    h.add_argument("--lr-nn", "--lr_nn", "--lr-nn-grid", dest="lr_nn", default="1e-4")
    h.add_argument("--beta", "--beta-grid", dest="beta", default="1.0")

    m = p.add_argument_group("model")
    m.add_argument("--a", type=int, default=2, help="smallest k searched")
    m.add_argument("--b", type=int, default=15, help="largest k searched")
    m.add_argument("--truth-k", "--truth_k", dest="truth_k", type=int, default=None,
                   help="ground-truth k; enables the *_t metrics. Must lie in [a, b].")
    m.add_argument("--epochs", type=int, default=200)
    m.add_argument("--m", type=int, default=1,
                   help="restarts per combination; the lowest-loss run is kept")
    m.add_argument("--batch-size", type=int, default=100)
    m.add_argument("--latent-dim", type=int, default=10)
    m.add_argument("--seed", type=int, default=42,
                   help="base random seed; restart j uses seed + j (default: 42)")
    m.add_argument("--decay-n", type=int, default=10)
    m.add_argument("--decay-nn", type=float, default=0.9)

    a = p.add_argument_group("autoencoder")
    a.add_argument("--ae-optimizer", choices=["rmsprop", "adam"], default="rmsprop",
                   help="adam pairs with --ae-loss binary_crossentropy to reproduce "
                        "old dmvae_rw runs")
    a.add_argument("--ae-loss", choices=["mean_squared_error", "binary_crossentropy"],
                   default="mean_squared_error")
    a.add_argument("--clipnorm", type=float, default=None,
                   help="gradient clipnorm for the DMVAE optimizer; old dmvae_rw used 5")
    a.add_argument("--ae-path", "--ae_path", dest="ae_path", default="",
                   help="reuse this saved autoencoder instead of training one")
    a.add_argument("--reuse-ae", "--reuse_ae", dest="reuse_ae", action="store_true",
                   help="reuse an existing ae_sim in the output directory. Off by "
                        "default: weights are never reused implicitly.")
    a.add_argument("--no-pretrain", action="store_true")
    return p


def _init_tensorflow_hpc():
    """Set TensorFlow environment defaults before the training stack loads."""
    # TensorFlow 2.15 already bundles legacy Keras. Forcing this flag there makes
    # ``tensorflow.keras`` unavailable when the separate tf-keras package is absent.
    # TensorFlow 2.16+ uses Keras 3 by default, so select tf-keras only on those
    # versions and continue to respect an explicit user setting.
    if "TF_USE_LEGACY_KERAS" not in os.environ:
        try:
            version = importlib.metadata.version("tensorflow")
        except importlib.metadata.PackageNotFoundError:
            version = importlib.metadata.version("tensorflow-macos")
        match = re.match(r"^(\d+)\.(\d+)", version)
        if match and tuple(map(int, match.groups())) >= (2, 16):
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


def _configure_tensorflow(tf):
    """Apply runtime settings after UMAP has loaded its LLVM dependencies."""
    try:
        tf.config.threading.set_inter_op_parallelism_threads(
            int(os.environ["TF_NUM_INTEROP_THREADS"]))
        tf.config.threading.set_intra_op_parallelism_threads(
            int(os.environ["TF_NUM_INTRAOP_THREADS"]))
    except Exception:
        pass
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def _validate_args(parser, args):
    """Reject invalid configurations before importing the training stack."""
    if args.a < 1 or args.b < args.a:
        parser.error("cluster bounds must satisfy 1 <= a <= b")
    if args.start >= args.end:
        parser.error("--start must be smaller than --end")
    for name in ("epochs", "m", "batch_size", "latent_dim", "decay_n"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.truth_k is not None and not args.a <= args.truth_k <= args.b:
        parser.error("--truth-k must lie inside [a, b]")


def _load_dataset(x_path, y_path, *, max_k):
    """Load and validate one cell-by-gene matrix and its reference labels."""
    import numpy as np

    X = np.loadtxt(x_path, dtype=np.float32, ndmin=2)
    labels = np.loadtxt(y_path, dtype=float, ndmin=1).reshape(-1)

    if X.shape[0] != labels.size:
        raise ValueError(
            f"row mismatch: {x_path!r} has {X.shape[0]} cells but "
            f"{y_path!r} has {labels.size} labels"
        )
    if X.shape[0] < max_k:
        raise ValueError(
            f"{X.shape[0]} cells are insufficient for b={max_k}; "
            "Gaussian-mixture initialization needs at least b cells"
        )
    if X.shape[1] < 1:
        raise ValueError("the expression matrix must contain at least one feature")
    if not np.all(np.isfinite(X)):
        raise ValueError("the expression matrix contains NaN or infinite values")
    if np.any(X < -1e-7) or np.any(X > 1 + 1e-7):
        raise ValueError("the expression matrix must be min-max scaled to [0, 1]")
    if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
        raise ValueError("reference labels must be finite integers")

    return X, labels.astype(np.int64)


def _slug(x):
    text = f"{x:.0e}" if isinstance(x, float) and 0 < x < 1e-3 else str(x)
    return re.sub(r"[^A-Za-z0-9_+=-]+", "_", text)


def datasets(args):
    """Yield (name, x_path, y_path). A {i} in the filename makes it a series."""
    if "{i}" in args.input_file or "{i}" in args.meta_file:
        for i in range(args.start, args.end):
            yield (f"sim{i}",
                   os.path.join(args.input_datafile, args.input_file.format(i=i)),
                   os.path.join(args.input_datafile, args.meta_file.format(i=i)))
    else:
        yield ("",
               os.path.join(args.input_datafile, args.input_file),
               os.path.join(args.input_datafile, args.meta_file))


def combinations(args):
    """Yield (name, dict of hyperparameters). One value each means a single run."""
    grids = {
        "ae_lr": _num_list(args.ae_lr, float),
        "ae_epoch": _num_list(args.ae_epoch, int),
        "lr_nn": _num_list(args.lr_nn, float),
        "beta": _num_list(args.beta, float),
    }
    single = all(len(v) == 1 for v in grids.values())
    for values in itertools.product(*grids.values()):
        combo = dict(zip(grids.keys(), values))
        name = "" if single else "aelr_{}_aep_{}_lrnn_{}_beta_{}".format(
            *(_slug(combo[k]) for k in ("ae_lr", "ae_epoch", "lr_nn", "beta")))
        yield name, combo


def _validated_combinations(parser, args):
    """Parse the hyperparameter grid and enforce meaningful numeric ranges."""
    try:
        parsed = list(combinations(args))
    except ValueError as exc:
        parser.error(f"invalid hyperparameter grid: {exc}")
    if not parsed:
        parser.error("each hyperparameter grid must contain at least one value")
    for _, combo in parsed:
        if combo["ae_lr"] <= 0 or combo["lr_nn"] <= 0:
            parser.error("learning rates must be positive")
        if combo["ae_epoch"] < 1:
            parser.error("--ae-epoch values must be at least 1")
        if combo["beta"] < 0:
            parser.error("--beta values must be non-negative")
    if not 0 < args.decay_nn <= 1:
        parser.error("--decay-nn must lie in (0, 1]")
    if args.clipnorm is not None and args.clipnorm <= 0:
        parser.error("--clipnorm must be positive")
    return parsed


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    combo_grid = _validated_combinations(parser, args)
    _init_tensorflow_hpc()

    # Imported after parse_args so --help does not load TensorFlow.
    from train import TrainConfig, run_training, save_run
    import tensorflow as tf

    _configure_tensorflow(tf)

    out_hyp = args.output_hyp or args.output_base_path
    os.makedirs(out_hyp, exist_ok=True)
    logfile = open(os.path.join(out_hyp, "callback.log"), "a", buffering=1)

    failures = []
    try:
        for ds_name, x_path, y_path in datasets(args):
            X, Y = _load_dataset(x_path, y_path, max_k=args.b)

            for combo_name, combo in combo_grid:
                out_dir = os.path.join(args.output_base_path, combo_name, ds_name)
                label = "/".join(p for p in (combo_name, ds_name) if p) or "run"
                print(f"=== {label}: {X.shape[0]} cells x {X.shape[1]} features")

                cfg = TrainConfig(
                    a=args.a, b=args.b, truth_k=args.truth_k, epochs=args.epochs, m=args.m,
                    batch_size=args.batch_size, latent_dim=args.latent_dim,
                    seed=args.seed,
                    decay_n=args.decay_n, decay_nn=args.decay_nn,
                    ae_optimizer=args.ae_optimizer, ae_loss=args.ae_loss,
                    clipnorm=args.clipnorm, ae_path=args.ae_path,
                    reuse_ae=args.reuse_ae, pretrain=not args.no_pretrain,
                    select=args.select, training_log=args.legacy_artifacts,
                    **combo)
                try:
                    res = run_training(X, Y, cfg, out_dir, logfile=logfile)
                except Exception as exc:                      # one bad combo must not
                    print(f"!!! {label} failed: {exc}")       # abandon the whole sweep
                    failures.append(label)
                    continue
                if res is None:
                    failures.append(label)
                    continue
                save_run(out_dir, X, Y, res,
                         multi_resolution=args.multi_resolution,
                         legacy_artifacts=args.legacy_artifacts)
                print(f"=== {label}: ARI={res['final']['ari']:.4f} "
                      f"NMI={res['final']['nmi']:.4f} K={res['final']['k_distinct']}")
    finally:
        logfile.close()

    if failures:
        print(f"\n{len(failures)} run(s) produced no results: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
