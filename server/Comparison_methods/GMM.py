import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import RMSprop
from sklearn.mixture import GaussianMixture as GMM
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import gc
import os
import umap
import argparse
import json
os.environ.setdefault("NUMBA_CACHE_DIR",
                      f"/tmp/numba_cache_{os.environ.get('SLURM_JOB_ID','local')}")
parser = argparse.ArgumentParser()
# Network structure
intermediate_dim = [500, 500, 2000]
batch_size = 100 
latent_dim = 10

parser.add_argument("--n_centroid", type=int, default=4, help="Number of GMM components (clusters)")
parser.add_argument("--n_repeat", type=int, default=100,help="How many independent repetitions to run")
parser.add_argument("--ae_lr", type=float, default=1e-6,help="Learning rate for SAE optimizer")
parser.add_argument("--ae_epoch", type=int, default=15,help="Epochs for SAE pretraining")
parser.add_argument("--input_datafile", type=str, required=True,help="Directory containing pbmc_norm.txt and pbmc_meta.txt")
parser.add_argument("--input_X", type=str, required=True, help="Directory containing pbmc_norm.txt and pbmc_meta.txt")
parser.add_argument("--input_Y", type=str, required=True,help="Directory containing pbmc_norm.txt and pbmc_meta.txt")
parser.add_argument("--output_base_path", type=str, required=True,help="Directory to write outputs (plots, accuracy txt)")

args = parser.parse_args()
n_centroid      = args.n_centroid
n_repeat        = args.n_repeat
ae_lr           = args.ae_lr
ae_epoch        = args.ae_epoch
input_datafile  = args.input_datafile.rstrip("/") + "/"
output_base_path = args.output_base_path.rstrip("/") + "/"

hyperparams = {
    "input_directory": input_datafile,
    "batch_size": batch_size,
    "ae_epochs": ae_epoch,
    "ae_learning_rate": ae_lr,
    "latent_dim": latent_dim,
    "n_repeats": n_repeat,
    "optimizer": "RMSprop",
    "layers": intermediate_dim,
    "n_centroid": n_centroid,
}
with open(os.path.join(output_base_path, "hyperparameters.json"), "w") as f:
    json.dump(hyperparams, f, indent=4)

def cluster_acc(Y_pred, Y):
    """Calculate clustering accuracy"""
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / Y_pred.size


os.makedirs(output_base_path, exist_ok=True)

# Load the same dataset once
X = np.loadtxt(input_datafile + args.input_X)  #"pbmc_norm_500.txt"
X[np.isnan(X)] = 0
original_dim = X.shape[1]
Y = np.loadtxt(input_datafile + args.input_Y).astype(int) #"pbmc_meta.txt"
all_acc = []
all_rep_losses = [] 

best_loss = float("inf")
best_info = {
    "rep": None,
    "loss": None,
    "acc": None,
    "umap": None,
    "y_pred": None,
    "plot_path": None
}

cmap = cm.get_cmap("tab10", n_centroid)
for i in range(1, n_repeat + 1):
    print(f"Processing repetition {i}...")

    gc.collect()

    ####### SAE model setup #######
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    latent = Dense(latent_dim, activation='relu')(h)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)
    encoder_sae = Model(x, latent, name="encoder")
    SAE = Model(x, x_decoded_mean)

    # Compile SAE
    rmsprop = RMSprop(learning_rate=ae_lr, clipnorm=5)
    SAE.compile(optimizer=rmsprop, loss='mean_squared_error')

    # Train
    history = SAE.fit(
        X, X,
        epochs=ae_epoch,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X, X),
        verbose=0
    )

    # Choose the metric: minimal validation loss across epochs (fallback to training loss)
    loss_seq = history.history.get("val_loss", history.history.get("loss"))
    rep_loss = float(np.min(loss_seq))
    all_rep_losses.append(rep_loss)

    # Encode -> cluster
    ae_zmean = encoder_sae.predict(X, verbose=0)
    g = GMM(n_components=n_centroid, covariance_type='diag', random_state=42)
    g.fit(ae_zmean)
    Y_pred = g.predict(ae_zmean)

    # Accuracy
    acc = cluster_acc(Y_pred, Y)
    all_acc.append(acc)
    print(f"  Repetition {i}: min_loss={rep_loss:.6f}, acc={acc:.6f}")

    # If this repetition is the best so far, compute UMAP & save plot
    if rep_loss < best_loss:
        best_loss = rep_loss
        best_info["rep"] = i
        best_info["loss"] = rep_loss
        best_info["acc"] = float(acc)

        # UMAP for visualization (fix random_state for reproducibility)
        umap_reducer = umap.UMAP(n_components=2)
        X_umap = umap_reducer.fit_transform(ae_zmean)
        best_info["umap"] = X_umap
        best_info["y_pred"] = Y_pred

        # Save a single "best" plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=Y_pred, cmap=cmap, s=8
        )
        plt.legend(*scatter.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Clusters")
        plt.title(f'Best GMM Clustering (k={n_centroid}) — Rep {i}\n(min loss={rep_loss:.6f}, acc={acc:.4f})')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')

        best_plot_path = os.path.join(output_base_path, "best_gmm_umap.png")
        plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        best_info["plot_path"] = best_plot_path
        
         # --- Combined figure with predicted (top) and true (bottom) ---
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))

        # Predicted clusters
        scatter1 = axes[0].scatter(
            X_umap[:, 0], X_umap[:, 1], c=Y_pred, cmap=cmap, s=4)
        axes[0].set_title(
            f'Predicted Clusters (best rep={i}, min loss={rep_loss:.6f}, acc={acc:.4f})'
        )
        axes[0].set_xlabel('UMAP Component 1')
        axes[0].set_ylabel('UMAP Component 2')
        axes[0].legend(*scatter1.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Clusters")

        # True labels
        n_true = len(np.unique(Y))
        cmap_true = cm.get_cmap("tab10", n_true)
        scatter2 = axes[1].scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=Y, cmap=cmap_true, s=4
        )
        axes[1].set_title("True Labels")
        axes[1].set_xlabel('UMAP Component 1')
        axes[1].set_ylabel('UMAP Component 2')
        axes[1].legend(*scatter2.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Clusters")

        plt.tight_layout()
        combined_plot_path = os.path.join(output_base_path, "best_umap_combined.png")
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Track path
        best_info["plot_combined_path"] = combined_plot_path


# Save arrays
acc_path   = os.path.join(output_base_path, "gmm_acc.txt")
loss_path  = os.path.join(output_base_path, "reconstruction_losses.txt")
stats_file = os.path.join(output_base_path, "gmm_acc_summary.txt")

np.savetxt(acc_path,  np.asarray(all_acc, dtype=np.float32))
np.savetxt(loss_path, np.asarray(all_rep_losses, dtype=np.float32))

acc_mean = float(np.mean(all_acc))
acc_var  = float(np.var(all_acc))   
acc_std  = float(np.std(all_acc))

# Summary (best by lowest loss)
with open(stats_file, "w") as f:
    f.write(f"Best repetition (by min loss): {best_info['rep']}\n")
    f.write(f"Best min loss: {best_info['loss']:.6f}\n")
    f.write(f"Accuracy at best repetition: {best_info['acc']:.6f}\n")
    f.write(f"Best plot saved to: {best_info['plot_path']}\n")

print(f"Saved accuracies to: {acc_path}")
print(f"Saved losses to:     {loss_path}")
print(f"Saved summary to:    {stats_file}")
print(f"Best repetition:     {best_info['rep']}, min loss={best_info['loss']:.6f}, acc={best_info['acc']:.6f}")
print(f"Best UMAP plot:      {best_info['plot_path']}")
print(f"Best combined UMAP plot (predicted above, true below): {best_info['plot_combined_path']}")
print(f"Mean accuracy across {n_repeat} reps: {acc_mean:.6f}")
print(f"Variance of accuracy: {acc_std:.6f}")
