import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from main import train, make_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convenience wrapper around main.py's train() for running scMAE on a single dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/sc_data/all_data/",
        help="Directory containing <dataset>.h5 files (same format expected by datasets.py).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name / prefix of the .h5 file (e.g., Pollen).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./res/",
        help="Directory where summary CSV files will be written.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="/data/sc_data/scMAE/",
        help="Root directory for embeddings, types, checkpoints, npz outputs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default matches main.py).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default matches main.py).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=4,
        help="Expected number of clusters; used by KMeans/leiden evaluation.",
    )
    parser.add_argument(
        "--data_dim",
        type=int,
        default=1000,
        help="Number of genes/features expected by the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; if omitted a random seed will be drawn.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is None:
        seed = random.randint(1, 100)
    else:
        seed = args.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.makedirs(args.results_dir, exist_ok=True)
    save_path = make_dir(os.path.join(args.save_root, f"seed_{seed}"), args.dataset)

    train_args = {
        "num_workers": 4,
        "paths": {"data": args.data_dir, "results": args.results_dir},
        "batch_size": args.batch_size,
        "data_dim": args.data_dim,
        "n_classes": args.n_classes,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "learning_rate": args.learning_rate,
        "latent_dim": 32,
        "save_path": save_path,
    }

    print("Running scMAE with configuration:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    print(f"  seed: {seed}")

    results = train(train_args)
    if results:
        summary_path = os.path.join(
            args.results_dir, f"scmae_summary_{args.dataset}_seed_{seed}.csv"
        )
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Saved summary metrics to {summary_path}")
    else:
        print(
            "No evaluation results were produced (e.g., epochs <= 80). "
            "Increase epochs to at least 81 to trigger evaluation."
        )


if __name__ == "__main__":
    main()


