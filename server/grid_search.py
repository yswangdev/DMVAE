import tensorflow as tf
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import RMSprop, Adam
import umap
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tensorflow.keras.callbacks import Callback
import gzip
from six.moves import cPickle
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from sklearn.mixture import GaussianMixture as GMM
import os
import gc
import json
import argparse
from itertools import product
import re
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import traceback

DISTINCT_COLORS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
    '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
    '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02', '#7570b3',
]

def _get_colors_cmap(labels):
    uniq = np.unique(labels)
    n = len(uniq)
    colors = DISTINCT_COLORS[:n] if n <= len(DISTINCT_COLORS) else plt.cm.tab20(np.linspace(0, 1, n))
    label_to_idx = {u: i for i, u in enumerate(uniq)}
    cvec = np.array([label_to_idx[l] for l in labels])
    cmap = ListedColormap(colors)
    return cvec, cmap, uniq, colors

def _add_labels(ax, xy, labels, colors_by_label, fontsize=9):
    xy, labels = np.asarray(xy), np.asarray(labels)
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() == 0:
            continue
        x, y = np.median(xy[mask], axis=0)
        ax.text(x, y, str(lab), fontsize=fontsize, fontweight='bold', ha='center', va='center',
                color='black', path_effects=[pe.withStroke(linewidth=2, foreground='white')])

class SkipCombo(Exception):
    pass

def safe_savetxt(path, arr):
    """Save only if arr is 1D/2D and non-empty."""
    if arr is None:
        print(f"Skip saving {os.path.basename(path)}: None")
        return
    a = np.asarray(arr)
    if a.size == 0:
        print(f"Skip saving {os.path.basename(path)}: empty")
        return
    if a.ndim == 0:
        print(f"Skip saving {os.path.basename(path)}: scalar (0-D)")
        return
    if a.ndim == 1:
        a = a.reshape(1, -1)
    np.savetxt(path, a)


def _parse_num_list(s, typ=float):
    """
    Accepts '1e-3,1e-4,1e-5' or '1e-3 1e-4 1e-5' (any mix of commas/spaces).
    Returns a list of numbers cast by typ (float or int).
    """
    if isinstance(s, list):  # (not used here, but future-proof)
        s = ",".join(s)
    parts = [p for p in re.split(r"[,\s]+", str(s).strip()) if p]
    return [typ(p) for p in parts]

parser = argparse.ArgumentParser()
parser.add_argument("--ae-lr-grid",      type=str, default="1e-3,1e-4,1e-5",
                    help="Comma/space-separated AE learning rates.")
parser.add_argument("--ae-epoch-grid",   type=str, default="10,15,20",
                    help="Comma/space-separated AE epochs (ints).")
parser.add_argument("--lr-nn-grid",      type=str, default="1e-3,1e-4,1e-5",
                    help="Comma/space-separated main NN learning rates.")
parser.add_argument("--beta-grid",       type=str, default="0.01,0.1,0.5,1",
                    help="Comma/space-separated beta values for KL weight.")
parser.add_argument("--input-datafile",  type=str, default=None,
                    help="Directory containing input files.")
parser.add_argument("--results-base",    type=str, default=None,
                    help="Base directory to write results.")
parser.add_argument("--grid-base",       type=str, default=None,
                    help="Directory for this grid run.")
parser.add_argument("--input-file",       type=str, default=None,
                    help="input file name.")    
parser.add_argument(
    "--ae_path",
    type=str,
    default="",    # means "decide automatically"
    help="Path to pre-trained AE model. "
         "If empty, use <output_datafile>/ae_sim, training if not present."
)
parser.add_argument("--meta-file",       type=str, default=None,
                    help="meta file name.")   
parser.add_argument("--truth-k",       type=int, default=None,
                    help="true number of clusters.")  
parser.add_argument("--a",       type=int, default=2,
                    help="lowe bound of number of clusters.") 
parser.add_argument("--b",       type=int, default=15,
                    help="upper bound of number of clusters")     
parser.add_argument("--epochs",       type=int, default=200,
                    help="epochs")   

args, _unknown = parser.parse_known_args()

# Convert to your variables
AE_LR_GRID    = _parse_num_list(args.ae_lr_grid, typ=float)
AE_EPOCH_GRID = _parse_num_list(args.ae_epoch_grid, typ=int)
LR_NN_GRID    = _parse_num_list(args.lr_nn_grid, typ=float)
BETA_GRID     = _parse_num_list(args.beta_grid, typ=float)

input_datafile = args.input_datafile
RESULTS_BASE   = args.results_base
GRID_BASE      = args.grid_base or os.path.join(RESULTS_BASE, "grid_single")
os.makedirs(GRID_BASE, exist_ok=True)

# --------------- Hyperparams & paths ---------------
intermediate_dim = [500, 500, 2000]
batch_size = 100
latent_dim = 10
decay_n, decay_nn, decay_gmm, alpha = 10, 0.9, 0.9, 1
ispretrain = True
truth_k = args.truth_k
epochs=args.epochs

start = 1
end = 2
m = 1 


def slug(x):
    if isinstance(x, float):
        x = f"{x:.0e}" if x < 1e-3 else str(x)
    return re.sub(r'[^A-Za-z0-9_+=-]+', '_', str(x))

# Data
x_t = np.loadtxt(os.path.join(input_datafile, args.input_file))
x_t[np.isnan(x_t)] = 0
global X, Y, original_dim
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt(os.path.join(input_datafile, args.meta_file)).astype(int)

# ---- trackers (globals) ----
best_loss = float('inf')
best_embedding = None
best_loss_curve = None
best_acc = None
best_assign_c = None
best_k = None
best_acc_t = None
best_assign_all = None
best_acc_all = None
best_z_mean = None
recon_loss = None
kl_loss_hist = None

all_loss, all_accuracy, all_accuracy_t, all_k = [], [], [], []

best_ari = None
best_nmi = None
best_ari_t = None
best_nmi_t = None
best_ari_all = None
best_nmi_all = None

all_ari, all_ari_t = [], []
all_nmi, all_nmi_t = [], []

# ----- Priors -----
a = args.a
b = args.b
# Is truth_k usable?
TRUTH_IN_RANGE = (truth_k is not None) and (a <= truth_k <= b)

def p_k_dist(priorDist):
    if priorDist == "uniform":
        p_k = 1 / (b - a + 1)  # scalar; broadcasts fine
        return p_k
    if priorDist == "poisson":
        k_values = np.arange(a, b + 1)
        poisson_pmf = np.exp(-10) * np.power(10, k_values) / np.array([math.factorial(k) for k in k_values])
        p_k = poisson_pmf / np.sum(poisson_pmf)
        return np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones((100, b - a + 1, b))
    if priorDist == "geometric":
        k_values = np.arange(a, b + 1)
        geometric_pmf = np.power(0.5, k_values) / 0.5
        p_k = geometric_pmf / np.sum(geometric_pmf)
        return np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones((100, b - a + 1, b))
    raise ValueError("Unknown priorDist")

p_k = p_k_dist("uniform")

def bad_pk(pk):
    """Return True if pk is None/empty/has NaNs/inf."""
    if pk is None:
        return True
    pk = np.asarray(pk)
    if pk.size == 0:
        return True
    if not np.all(np.isfinite(pk)):
        return True
    return False

# ----- Layers / utils -----
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_now = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_now, tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

from scipy.optimize import linear_sum_assignment

def cluster_acc(Y_pred, Y_true):
    assert Y_pred.size == Y_true.size
    D = max(Y_pred.max(), Y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(row_ind, col_ind)) / Y_pred.size, list(zip(row_ind, col_ind))

def gmmpara_init():
    theta_init, u_init, lambda_init = [], [], []
    for n_centroid in range(a, b + 1):
        theta_init.append(np.ones(n_centroid) / n_centroid)
        u_init.append(np.zeros((latent_dim, n_centroid)))
        lambda_init.append(np.ones((latent_dim, n_centroid)))

    theta_init_padded = np.array([np.pad(t, (0, b - len(t)), 'constant', constant_values=1e-10) for t in theta_init])
    u_init_padded = np.array([np.pad(u, ((0, 0), (0, b - u.shape[1])), 'constant', constant_values=1e-10) for u in u_init])
    lambda_init_padded = np.array([np.pad(l, ((0, 0), (0, b - l.shape[1])), 'constant', constant_values=1) for l in lambda_init])

    theta_p = tf.Variable(theta_init_padded, trainable=True, dtype=tf.float32, name="pi")
    u_p     = tf.Variable(u_init_padded,     trainable=True, dtype=tf.float32, name="u")
    lambda_p= tf.Variable(lambda_init_padded,trainable=True, dtype=tf.float32, name="lambda")
    return theta_p, u_p, lambda_p

class get_gamma(Layer):
    def call(self, inputs):
        temp_Z_set = []
        batch_now = tf.shape(inputs)[0]
        for n_centroid in range(a, b + 1):
            Z_temp = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, n_centroid])
            temp_Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
            temp_Z_set.append(temp_Z_padded)
        temp_Z = tf.cast(tf.stack(temp_Z_set, axis=1), tf.float32)
        temp_u_tensor3 = tf.repeat(tf.expand_dims(u_p, 0), batch_now, axis=0)
        temp_lambda_tensor3 = tf.repeat(tf.expand_dims(lambda_p, 0), batch_now, axis=0)
        temp_theta_tensor3 = tf.expand_dims(tf.expand_dims(theta_p, 0), 2) * tf.ones((batch_now, b - a + 1, latent_dim, b))

        temp_p_c_z = tf.exp(
            tf.reduce_sum((tf.math.log(temp_theta_tensor3) - 0.5 * tf.math.log(2 * np.pi * temp_lambda_tensor3) -
                           tf.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)), axis=2)
        ) + 1e-10
        gamma = temp_p_c_z / tf.reduce_sum(tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
        p_k_z = tf.reduce_sum(gamma, axis=-1)
        p_c_z = gamma / tf.reduce_sum(gamma, axis=-1, keepdims=True)
        return p_k_z, p_c_z

class DMVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        if np.isnan(theta_p).any() or np.isnan(u_p).any() or np.isnan(lambda_p).any():
            raise ValueError("Initial parameters contain NaNs")

        self.theta_p = theta_p
        self.u_p = u_p
        self.lambda_p = lambda_p

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            batch_now = tf.shape(z)[0]

            Z_set, z_mean_set, z_log_var_set = [], [], []
            for n_centroid in range(a, b + 1):
                Z_temp = tf.tile(tf.expand_dims(z, axis=2), [1, 1, n_centroid])
                Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
                Z_set.append(Z_padded)

                z_mean_temp = tf.tile(tf.expand_dims(z_mean, axis=2), [1, 1, n_centroid])
                z_mean_padded = tf.pad(z_mean_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
                z_mean_set.append(z_mean_padded)

                z_log_var_temp = tf.tile(tf.expand_dims(z_log_var, axis=2), [1, 1, n_centroid])
                z_log_var_padded = tf.pad(z_log_var_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=-(1e+10))
                z_log_var_set.append(z_log_var_padded)

            Z = tf.cast(tf.stack(Z_set, axis=1), tf.float32)
            z_mean_t = tf.cast(tf.stack(z_mean_set, axis=1), tf.float32)
            z_log_var_t = tf.cast(tf.stack(z_log_var_set, axis=1), tf.float32)

            u_tensor3 = tf.repeat(tf.expand_dims(self.u_p, 0), batch_now, axis=0)
            lambda_tensor3 = tf.repeat(tf.expand_dims(self.lambda_p, 0), batch_now, axis=0)
            theta_tensor3 = tf.expand_dims(tf.expand_dims(self.theta_p, 0), 2) * tf.ones((batch_now, b - a + 1, latent_dim, b))

            p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) -
                                 K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=2)) + 1e-10

            gamma = p_c_z / tf.reduce_sum(tf.reduce_sum(p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
            gamma_t = tf.repeat(tf.expand_dims(gamma, 2), latent_dim, axis=2)

            reconstruction_loss = alpha * original_dim * tf.keras.losses.mean_squared_error(data, reconstruction)
            k13 = K.exp(z_log_var_t) / lambda_tensor3
            kl_loss = K.sum(0.5 * gamma_t * (latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + k13 +
                                             K.square(z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2, 3)) \
                      - 0.5 * K.sum(z_log_var + 1, axis=-1) \
                      - K.sum(K.sum(K.log(tf.repeat(tf.expand_dims(self.theta_p, 0), repeats=batch_now, axis=0) *
                                                  tf.expand_dims(p_k, 0) + 1e-10) * gamma, axis=-1), axis=1) \
                      + K.sum(K.sum(K.log(gamma) * gamma, axis=-1), axis=1)

            total_loss = reconstruction_loss + beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

def load_pretrain_weights(dmvae):
    ae = load_model(args.ae_path)
    # encoder weights
    dmvae.encoder.layers[1].set_weights(ae.layers[1].get_weights())
    dmvae.encoder.layers[2].set_weights(ae.layers[2].get_weights())
    dmvae.encoder.layers[3].set_weights(ae.layers[3].get_weights())
    dmvae.encoder.layers[4].set_weights(ae.layers[4].get_weights())
    # decoder weights
    dmvae.decoder.layers[-1].set_weights(ae.layers[-1].get_weights())
    dmvae.decoder.layers[-2].set_weights(ae.layers[-2].get_weights())
    dmvae.decoder.layers[-3].set_weights(ae.layers[-3].get_weights())
    dmvae.decoder.layers[-4].set_weights(ae.layers[-4].get_weights())

    zmean_model = Model(inputs=dmvae.encoder.input, outputs=dmvae.encoder.output[0])
    sample = zmean_model.predict(X, batch_size=batch_size, verbose=0)

    for n_centroid in range(a, b + 1):
        g = GMM(n_components=n_centroid, covariance_type='diag', random_state=42, reg_covar=1e-3,      # <--- add this
    n_init=5)
        g.fit(sample)
        means_reshaped = tf.transpose(tf.convert_to_tensor(g.means_, dtype=tf.float32))
        means_padded = tf.pad(means_reshaped, [[0, 0], [0, b - n_centroid]], constant_values=1e-10)
        covariances_reshaped = tf.transpose(tf.convert_to_tensor(g.covariances_, dtype=tf.float32))
        covariances_padded = tf.pad(covariances_reshaped, [[0, 0], [0, b - n_centroid]], constant_values=1)
        dmvae.u_p[n_centroid - a, :, :].assign(means_padded)
        dmvae.lambda_p[n_centroid - a, :, :].assign(covariances_padded)
    print('Pretrain weights loaded!')
    return dmvae

def lr_decay():
    # If you later add a GMM optimizer, handle it here; for now only nn is used
    current_lr_nn = rmsprop_nn.learning_rate.numpy()
    new_lr_nn = max(current_lr_nn * decay_nn, 1e-7)
    rmsprop_nn.learning_rate.assign(new_lr_nn)
    print('lr_nn: %f' % rmsprop_nn.learning_rate.numpy())

class EpochBegin(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip = False
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % decay_n == 0 and epoch != 0:
            lr_decay()

        p_k_z, p_c_z = gamma_output.predict(X, batch_size=batch_size, verbose=0)

        k = np.argmax(tf.reduce_sum(p_k_z, axis=0), axis=-1)
        pk = tf.reduce_sum(p_k_z, axis=0)  # total responsibility per k (sums to N), not a probability
        k_list.append(k)
        posteriorK.append(pk.numpy())

        k_order = tf.argsort(tf.reduce_sum(p_k_z, axis=0), direction='DESCENDING')
        k_order_list.append(k_order.numpy())
        p_c_label = p_c_z[:, k, :]
        assign_c = np.argmax(p_c_label, axis=1)
        assign.append(assign_c)
        acc = cluster_acc(assign_c, Y)
        accuracy.append(acc[0])

        # Selected-k metrics (always)
        ari.append(ARI(Y, assign_c))
        nmi.append(NMI(Y, assign_c, average_method='arithmetic'))
        
        # Truth-k metrics (only if in range)
        if TRUTH_IN_RANGE:
            p_truth_label = p_c_z[:, truth_k - a, :]
            assign_truth = np.argmax(p_truth_label, axis=1)
            acc_t = cluster_acc(assign_truth, Y)
            accuracy_t.append(acc_t[0])
            ari_t.append(ARI(Y, assign_truth))
            nmi_t.append(NMI(Y, assign_truth, average_method='arithmetic'))
        else:
            # keep arrays aligned across epochs
            accuracy_t.append(np.nan)
            ari_t.append(np.nan)
            nmi_t.append(np.nan)

        for cl in range(0, b - a + 1):
            k_value = cl + a
            label = p_c_z[:, cl, :]
            assign_all[k_value] = np.argmax(label, axis=1)
            acc_all[k_value] = cluster_acc(assign_all[k_value], Y)[0]

            # --- Per-k ARI/NMI (mirrors acc_all) ---
        for k_value in range(a, b + 1):
            ari_all[k_value] = ARI(Y, assign_all[k_value])
            nmi_all[k_value] = NMI(Y, assign_all[k_value], average_method='arithmetic')

        if epoch > 0:
            print('k_order:', k_order.numpy())
            print('acc:%0.8f' % acc[0])
            print("pk", pk.numpy())
            if bad_pk(pk.numpy()):
                print("pk is invalid (NaN/Inf/empty). Skipping this hyperparameter combo.\n")
                self.skip = True
                self.model.stop_training = True
                return

    def on_epoch_end(self, epoch, logs=None):
        # re-pad upper triangle to constants
        for i in range(b - a):
            self.model.theta_p[i, i + a:b].assign(tf.constant(1e-10, shape=[b - (i + a)], dtype=tf.float32))
            self.model.u_p[i, :, i + a:b].assign(tf.constant(1e-10, shape=[latent_dim, b - (i + a)], dtype=tf.float32))
            self.model.lambda_p[i, :, i + a:b].assign(tf.constant(1, shape=[latent_dim, b - (i + a)], dtype=tf.float32))

def run_dmvae_for_current_hparams(output_base_path_for_combo):
    import time
    global best_loss, best_z_mean, best_loss_curve, best_acc, best_assign_c, best_k, best_acc_t
    global best_assign_all, best_acc_all, best_embedding, recon_loss, kl_loss_hist
    global all_loss, all_accuracy, all_accuracy_t, all_k
    global theta_p, u_p, lambda_p, gamma_output, rmsprop_nn
    global k_list, k_order_list, accuracy, accuracy_t, assign, assign_all, acc_all
    global ari, nmi, ari_t, nmi_t, ari_all, nmi_all
    global posteriorK, all_pk, best_pk

    # --- RESET PER COMBO ---
    best_loss = float('inf')
    best_z_mean = None
    best_loss_curve = None
    best_acc = None
    best_assign_c = None
    best_k = None
    best_acc_t = None
    best_assign_all = {}
    best_acc_all = {}
    best_embedding = None
    recon_loss = None
    kl_loss_hist = None


    all_loss = []
    all_accuracy = []
    all_accuracy_t = []
    all_k = []

    # --- RESET PER COMBO ---
    best_ari = None
    best_nmi = None
    best_ari_t = None
    best_nmi_t = None
    best_ari_all = {}
    best_nmi_all = {}

    all_ari = []
    all_ari_t = []
    all_nmi = []
    all_nmi_t = []
    all_pk = []
    best_pk = None
    # -----------------------
    # -----------------------

    start_time = time.time()
    output_datafile = output_base_path_for_combo
    os.makedirs(output_datafile, exist_ok=True)
    last_ae_history = None

    # local SAE path within this combo
    ae_path_default = os.path.join(output_datafile, "ae_sim")
    # Decide which AE path to use
    if args.ae_path:  # user provided something via --ae_path
        ae_path = args.ae_path
        print(f"[AE] Using user-specified AE path: {ae_path}", flush=True)
        train_sae = False   # assume it already exists; we won't train here
    else:
        ae_path = ae_path_default
        if os.path.exists(ae_path):
            print(f"[AE] Found existing AE at {ae_path}, will reuse it.", flush=True)
            train_sae = False
        else:
            print(f"[AE] No AE found at {ae_path}, will train a new SAE.", flush=True)
            train_sae = True
    
    # Remember this path for later loading
    args.ae_path = ae_path
    
    for j in range(0, m):
        K.clear_session(); gc.collect()

        # ===== SAE =====
        x_in = Input(shape=(original_dim,))
        h = Dense(intermediate_dim[0], activation='relu')(x_in)
        h = Dense(intermediate_dim[1], activation='relu')(h)
        h = Dense(intermediate_dim[2], activation='relu')(h)
        latent = Dense(latent_dim, activation='relu')(h)
        h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent)
        h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
        h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
        x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)
        encoder_sae = Model(x_in, latent, name="encoder")
        SAE = Model(x_in, x_decoded_mean)
        SAE.compile(optimizer=RMSprop(learning_rate=ae_lr, clipnorm=5), loss='mean_squared_error')
        history_ae = SAE.fit(X, X, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X), verbose=0)
        SAE.save(ae_path)
        last_ae_history = history_ae.history

        # ===== Save SAE latent space =====
        # Get latent codes from the trained SAE encoder
        ae_latent = encoder_sae.predict(X, batch_size=batch_size, verbose=0)

        # Save as .txt (human-readable) and .npy (fast/precise)
        np.savetxt(os.path.join(output_datafile, 'ae_latent.txt'), ae_latent)
        np.save(os.path.join(output_datafile, 'ae_latent.npy'), ae_latent)

        # Optional: quick 2D UMAP of AE latent space (colored by true labels)
        reducer_ae = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        ae_umap = reducer_ae.fit_transform(ae_latent)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        s = ax.scatter(ae_umap[:, 0], ae_umap[:, 1], c=Y, s=2)
        ax.set_title('SAE Latent UMAP (colored by true labels)', fontsize=16)
        ax.legend(*s.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
        plt.tight_layout()
        plt.savefig(os.path.join(output_datafile, 'umap_ae.png'), bbox_inches='tight')
        plt.savefig(os.path.join(output_datafile, 'ae.png'), bbox_inches='tight')
        plt.close(fig);
        plt.clf()

        # Optional: store the 2D UMAP coordinates too
        np.savetxt(os.path.join(output_datafile, 'ae_umap_2d.txt'), ae_umap)
        np.save(os.path.join(output_datafile, 'ae_umap_2d.npy'), ae_umap)

        # ===== DMVAE =====
        k_list, k_order_list, accuracy, accuracy_t, assign = [], [], [], [], []
        assign_all, acc_all = {}, {}
        ari, nmi, ari_t, nmi_t = [], [], [], []
        ari_all, nmi_all = {}, {}
        posteriorK = []

        theta_p, u_p, lambda_p = gmmpara_init()

        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim[0], activation='relu')(x)
        h = Dense(intermediate_dim[1], activation='relu')(h)
        h = Dense(intermediate_dim[2], activation='relu')(h)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        z = Sampling()([z_mean, z_log_var])
        h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
        h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
        h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
        x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)

        p_k_z, p_c_z = get_gamma()(z)
        gamma_output = Model(inputs=x, outputs=[p_k_z, p_c_z])
        encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
        decoder = Model(inputs=z, outputs=x_decoded_mean, name="decoder")
        dmvae = DMVAE(encoder, decoder)

        if ispretrain:
            dmvae = load_pretrain_weights(dmvae)

        rmsprop_nn = RMSprop(learning_rate=lr_nn)
        dmvae.compile(optimizer=rmsprop_nn)
        
        cb = EpochBegin()
        fitting = dmvae.fit(X, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[cb], verbose=0)
        if getattr(cb, "skip", False):
            print("Skipped saving artifacts for this combo due to invalid pk.")
            return

        last_loss = fitting.history['loss'][-1]
        last_recon_loss = fitting.history['reconstruction_loss'][-1]
        last_kl_loss = fitting.history['kl_loss'][-1]
        all_loss.append({'loss': float(last_loss), 'reconstruction_loss': float(last_recon_loss), 'kl_loss': float(last_kl_loss)})
        all_accuracy.append(accuracy[-1]); all_accuracy_t.append(accuracy_t[-1]); all_k.append(k_list[-1])
        all_ari.append(ari[-1])
        all_nmi.append(nmi[-1])
        all_ari_t.append(ari_t[-1])
        all_nmi_t.append(nmi_t[-1])
        posteriorK_arr = np.array(posteriorK)
        all_pk.append(posteriorK_arr[-1])

        z_mean_arr, _, _ = dmvae.encoder.predict(X, batch_size=batch_size, verbose=0)
        if last_loss < best_loss:
            best_loss = last_loss
            best_z_mean = z_mean_arr
            best_loss_curve = fitting.history['loss']
            recon_loss = fitting.history['reconstruction_loss']
            kl_loss_hist = fitting.history['kl_loss']
            best_acc = accuracy
            best_assign_c = assign[-1]
            best_k = k_list
            best_acc_t = accuracy_t
            best_assign_all = assign_all
            best_acc_all = acc_all
            best_ari = ari
            best_nmi = nmi
            best_ari_t = ari_t
            best_nmi_t = nmi_t
            best_acc_all = acc_all.copy()
            best_ari_all = ari_all.copy()
            best_nmi_all = nmi_all.copy()
            best_pk = posteriorK_arr[-1]
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            best_embedding = reducer.fit_transform(z_mean_arr)
        print(f"Finished iteration {j}...")

    # ===== Save artifacts =====
    safe_savetxt(os.path.join(output_datafile, 'z_mean.txt'), best_z_mean)
    safe_savetxt(os.path.join(output_datafile, 'DMVAE_loss.txt'), best_loss_curve)
    safe_savetxt(os.path.join(output_datafile, 'accuracy.txt'), np.array(best_acc))
    safe_savetxt(os.path.join(output_datafile, 'accuracy_t.txt'), np.array(best_acc_t))
    safe_savetxt(os.path.join(output_datafile, 'k.txt'), np.array(best_k))
    safe_savetxt(os.path.join(output_datafile, 'assign_c.txt'), best_assign_c)
    np.save(os.path.join(output_datafile, "best_acc_all.npy"), best_acc_all)
    with open(os.path.join(output_datafile, 'accuracies_all_k.json'), 'w') as f:
        json.dump({str(k): float(v) for k, v in best_acc_all.items()}, f, indent=2)
    if best_pk is not None:
        # best_pk is total responsibility per k (sums to N); save normalized so values are in [0,1]
        n_samples = best_z_mean.shape[0]
        pk_normalized = best_pk / np.maximum(n_samples, 1)
        np.savetxt(os.path.join(output_datafile, 'posteriorK_best.txt'), np.atleast_2d(best_pk))
        np.savetxt(os.path.join(output_datafile, 'posteriorK_best_normalized.txt'), np.atleast_2d(pk_normalized))
    if all_pk:
        np.save(os.path.join(output_datafile, 'posteriorK_all_iters.npy'), np.stack(all_pk))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    s1 = ax1.scatter(best_embedding[:, 0], best_embedding[:, 1], c=best_assign_c, s=1, cmap='viridis')
    ax1.set_title('predicted label', fontsize=24)
    lines = [
        f'Acc: {best_acc[-1]:.3f}',
        f'ARI: {best_ari[-1]:.3f}  NMI: {best_nmi[-1]:.3f}',
        f'k: {best_k[-1] + a}',
    ]
    # Only add truth-k line if it’s valid and not NaN
    if TRUTH_IN_RANGE and not np.isnan(best_acc_t[-1]):
        lines.insert(1, f'Acc@truthk: {best_acc_t[-1]:.3f}')
    
    ax1.text(
        0.95, 0.95, "\n".join(lines),
        transform=ax1.transAxes, fontsize=12, va='top', ha='right',
        bbox=dict(facecolor='white', alpha=0.8)
    )
    ax1.legend(*s1.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
    s2 = ax2.scatter(best_embedding[:, 0], best_embedding[:, 1], c=Y, s=1)
    ax2.set_title('True Label', fontsize=24)
    ax2.legend(*s2.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
    plt.tight_layout(); plt.savefig(os.path.join(output_datafile, "umap_DMVAE_best.png"), bbox_inches='tight')
    plt.close(fig); plt.clf()
    

    # ---- Per-k UMAP plots with ACC / ARI / NMI annotations ----
    for k in sorted(best_assign_all.keys()):
        pred_k = best_assign_all[k]
        acc_k = best_acc_all.get(k, None)
        ari_k = best_ari_all.get(k, None)
        nmi_k = best_nmi_all.get(k, None)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
        xy = best_embedding[:, 0], best_embedding[:, 1]
        xy = np.column_stack(xy)
    
        # Predicted
        cvec_p, cmap_p, uniq_p, cols_p = _get_colors_cmap(pred_k)
        ax1.scatter(xy[:, 0], xy[:, 1], c=cvec_p, s=1, cmap=cmap_p, vmin=-0.5, vmax=len(uniq_p) - 0.5)
        _add_labels(ax1, xy, pred_k, {u: cols_p[i] for i, u in enumerate(uniq_p)})
        ax1.set_title(f'Predicted Label (k={k})', fontsize=24)
        ax1.legend(handles=[Patch(facecolor=cols_p[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_p)],
                   loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
    
        lines = []
        if acc_k is not None: lines.append(f'ACC: {acc_k:.3f}')
        if ari_k is not None: lines.append(f'ARI: {ari_k:.3f}')
        if nmi_k is not None: lines.append(f'NMI: {nmi_k:.3f}')
        if lines:
            ax1.text(0.98, 0.02, '\n'.join(lines), transform=ax1.transAxes, ha='right', va='bottom',
                     fontsize=12, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    
        # True
        cvec_t, cmap_t, uniq_t, cols_t = _get_colors_cmap(Y)
        ax2.scatter(xy[:, 0], xy[:, 1], c=cvec_t, s=1, cmap=cmap_t, vmin=-0.5, vmax=len(uniq_t) - 0.5)
        _add_labels(ax2, xy, Y, {u: cols_t[i] for i, u in enumerate(uniq_t)})
        ax2.set_title('True Label', fontsize=24)
        ax2.legend(handles=[Patch(facecolor=cols_t[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_t)],
                   loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
    
        plt.tight_layout()
        plt.savefig(os.path.join(output_datafile, f"umap_DMVAE{k}.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)
        plt.clf()

    plt.figure(figsize=(8, 6))
    plt.plot(best_loss_curve, label='Total Loss')
    plt.plot(recon_loss, label='Reconstruction Loss')
    plt.plot(kl_loss_hist, label='KL Loss')
    plt.legend(); plt.title('dmvae Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.savefig(os.path.join(output_datafile, 'DMVAE_loss.png')); plt.clf()

    if last_ae_history:
        plt.figure(figsize=(6, 4))
        plt.plot(last_ae_history['loss'], label='AE loss')
        plt.legend(); plt.title('AE Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
        plt.savefig(os.path.join(output_datafile, 'ae_loss.png')); plt.close(); plt.clf()

    with open(os.path.join(output_datafile, 'all_loss.json'), 'w') as f:
        json.dump(all_loss, f, indent=4)
    np.savetxt(os.path.join(output_datafile, 'all_accuracy.txt'), np.array(all_accuracy))
    np.savetxt(os.path.join(output_datafile, 'all_k.txt'), np.array(all_k))
    np.savetxt(os.path.join(output_datafile, 'all_accuracy_t.txt'), np.array(all_accuracy_t))
    # Per-epoch (best run in this combo)
    np.savetxt(os.path.join(output_datafile, 'ari.txt'), np.array(best_ari))
    np.savetxt(os.path.join(output_datafile, 'nmi.txt'), np.array(best_nmi))
    np.savetxt(os.path.join(output_datafile, 'ari_t.txt'), np.array(best_ari_t))
    np.savetxt(os.path.join(output_datafile, 'nmi_t.txt'), np.array(best_nmi_t))

    # Per-k (best run in this combo)
    with open(os.path.join(output_datafile, 'ari_all.json'), 'w') as f:
        json.dump({int(k): float(v) for k, v in best_ari_all.items()}, f, indent=4)
    with open(os.path.join(output_datafile, 'nmi_all.json'), 'w') as f:
        json.dump({int(k): float(v) for k, v in best_nmi_all.items()}, f, indent=4)
    with open(os.path.join(output_datafile, "assignments_all_k.json"), "w") as f:
        serializable = {int(k): np.asarray(v).astype(int).tolist()
                        for k, v in best_assign_all.items()}
        json.dump(serializable, f, indent=2)

    # Cross-combo summaries (last-epoch of each run in this combo)
    np.savetxt(os.path.join(output_datafile, 'all_ari.txt'), np.array(all_ari))
    np.savetxt(os.path.join(output_datafile, 'all_nmi.txt'), np.array(all_nmi))
    np.savetxt(os.path.join(output_datafile, 'all_ari_t.txt'), np.array(all_ari_t))
    np.savetxt(os.path.join(output_datafile, 'all_nmi_t.txt'), np.array(all_nmi_t))

    end_time = time.time()
    np.savez(os.path.join(output_datafile, 'dmvae.npz'), ARI=np.array(best_ari), NMI=np.array(best_nmi), K=np.array(best_k), ACC=np.array(best_acc), Embedding=np.array(best_z_mean),
             Clusters=np.array(best_assign_c), Time_use=end_time - start_time)
    print(f"Total time: {end_time - start_time} seconds.")

# --------------- Grid search loop ---------------
for ae_lr_val, ae_epoch_val, lr_nn_val, beta_val in product(AE_LR_GRID, AE_EPOCH_GRID, LR_NN_GRID, BETA_GRID):
    ae_lr = ae_lr_val
    ae_epoch = ae_epoch_val
    lr_nn = lr_nn_val
    beta = beta_val

    combo_name = f"aelr_{slug(ae_lr)}_aep_{slug(ae_epoch)}_lrnn_{slug(lr_nn)}_beta_{slug(beta)}"
    combo_base = os.path.join(GRID_BASE, combo_name)
    os.makedirs(combo_base, exist_ok=True)

    with open(os.path.join(combo_base, "hyperparameters.json"), "w") as f:
        json.dump({
            "scenario": 4,
            "batch_size": batch_size,
            "ae epochs": ae_epoch,
            "ae learning rate": ae_lr,
            "epochs": epochs,
            "learning_rate": lr_nn,
            "latent_dim": latent_dim,
            "#pretrain": m,
            "optimizer": "RMSprop",
            "loss_function": "mean_squared_error",  # was 'binary_crossentropy'
            "layers": intermediate_dim,
            "beta": beta
        }, f, indent=4)

    print(f"\n=== Running combo: {combo_name} ===")
    run_dmvae_for_current_hparams(combo_base + os.sep)