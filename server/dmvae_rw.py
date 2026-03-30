import tensorflow as tf
import numpy as np
import pandas as pd
import keras.backend as K
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import RMSprop, Adam
from sklearn import metrics
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from keras.models import model_from_json
from tensorflow.keras.callbacks import Callback
import gzip
from six.moves import cPickle
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from sklearn.mixture import GaussianMixture as GMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def ARI(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)
def NMI(y_true, y_pred, average_method='arithmetic'):
    return normalized_mutual_info_score(y_true, y_pred, average_method=average_method)
import umap
import os
import gc
import argparse


intermediate_dim = [500, 500, 2000]
batch_size = 100
latent_dim = 10
decay_n, alpha = 10, 1
ispretrain = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr_nn', type=float, default=1e-6, help="DMVAE Learning rate")
parser.add_argument("--lr_gmm", type=float, default=1e-6, help="Learning rate for GMM")
parser.add_argument("--epochs", type=int, default=100, help="DMVAE epochs")
parser.add_argument('--ae_lr', type=float, default=1e-5, help="AE learning rate")
parser.add_argument('--ae_epoch', type=int, default=10, help="AE epoch")
parser.add_argument("--truth_k", type=int, required=True, help="Ground truth cluster count")
parser.add_argument("--input_datafile", type=str, required=True, help="Path to input data directory")
parser.add_argument("--input_file", type=str, required=True, help="input data")
parser.add_argument("--meta_file", type=str, required=True, help="meta data")
parser.add_argument("--output_datafile", type=str, required=True, help="Path for output results")
parser.add_argument("--output_hyp", type=str, required=True, help="Path for hyperparameter outputs")
parser.add_argument("--m", type=int, default=100, help="number of pretrain times")
parser.add_argument("--beta", type=float, default=1, help="beta from beta-vae")
parser.add_argument("--decay_nn", type=float, default=0.8, help="decay rate")
parser.add_argument("--decay_gmm", type=float, default=0.8, help="decay rate")
parser.add_argument("--a", type=int, default=2, help="min number of clusters")
parser.add_argument("--b", type=int, default=6, help="max number of clusters")
args = parser.parse_args()

import json

# Example hyperparameters
hyperparams = {
    "input directory" : args.input_datafile,
    "input file": args.input_file,
    "batch_size": batch_size,
    "ae epochs": args.ae_epoch,
    "ae learning rate": args.ae_lr,
    "epochs": args.epochs,
    "learning_rate": args.lr_nn,
    "latent_dim": latent_dim,
    "#pretrain": args.m,
    "optimizer": "RMSprop",
    "layers": intermediate_dim,
    "beta": args.beta,
    "decay_rate": args.decay_nn,
    "a": args.a,
    "b": args.b
}

# Save to a JSON file
with open(args.output_hyp + "hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=4)
    
    
x_t = np.loadtxt(args.input_datafile + args.input_file)
x_t[np.isnan(x_t)] = 0
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt(args.input_datafile + args.meta_file)
Y = Y.astype(int)

# Plot style (match dmvae_keras2 / grid_search)
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


def p_k_dist(priorDist):
    if priorDist == "uniform":
        p_k = 1/ (b - a + 1)
        p_k_expanded = p_k
    if priorDist == "poisson":
        k_values = np.arange(a, b + 1)
        poisson_pmf = np.exp(-10) * np.power(10, k_values) / np.array(
            [math.factorial(k) for k in k_values])
        normalization_factor = np.sum(poisson_pmf)
        p_k = poisson_pmf / normalization_factor
        p_k_expanded = np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones(
            (100, b - a + 1, b))
    if priorDist == "geometric":
        k_values = np.arange(a, b + 1)
        geometric_pmf = np.power(0.5, k_values) / 0.5
        normalization_factor = np.sum(geometric_pmf)
        p_k = geometric_pmf / normalization_factor
        p_k_expanded = np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones(
                (100, b - a + 1, b))
    return p_k_expanded

a = args.a
b = args.b
p_k = p_k_dist("uniform")

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_now = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_now, tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


########### cluster accuracy ###########

def cluster_acc(Y_pred, Y):
    # Function to calculate clustering accuracy
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    ind = list(zip(row_ind, col_ind))  # Convert the result into pairs of indices
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, ind

def calculate_metric(pred, label):
    # acc = np.round(cluster_acc(label, pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 4)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 4)

    return nmi, ari

########### gamma   ###########
def gmmpara_init():
    theta_init = []
    u_init = []
    lambda_init = []
    for n_centroid in range(a, b + 1):
        theta_init.append(np.ones(n_centroid) / n_centroid)
        u_init.append(np.zeros((latent_dim, n_centroid)))
        lambda_init.append(np.ones((latent_dim, n_centroid)))

    theta_init_padded = np.array(
        [np.pad(theta, (0, b - len(theta)), 'constant', constant_values=1e-10) for theta in theta_init])
    u_init_padded = np.array(
        [np.pad(u, ((0, 0), (0, b - u.shape[1])), 'constant', constant_values=1e-10) for u in u_init])
    lambda_init_padded = np.array(
        [np.pad(lambda_, ((0, 0), (0, b - lambda_.shape[1])), 'constant', constant_values=1) for lambda_ in
         lambda_init])

    theta_p = tf.Variable(theta_init_padded, trainable=True, dtype=tf.float32, name="pi")
    u_p = tf.Variable(u_init_padded, trainable=True, dtype=tf.float32, name="u")
    lambda_p = tf.Variable(lambda_init_padded, trainable=True, dtype=tf.float32, name="lambda")

    return theta_p, u_p, lambda_p


class get_gamma(Layer):
    def call(self, inputs):
        batch_now = tf.shape(inputs)[0]  # dynamic batch
        temp_Z_set = []
        for n_centroid in range(a, b + 1):
            Z_temp = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, n_centroid])
            temp_Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
            temp_Z_set.append(temp_Z_padded)

        temp_Z = tf.cast(tf.stack(temp_Z_set, axis=1), tf.float32)

        # repeat params along the CURRENT batch, not the global batch_size
        temp_u_tensor3 = tf.repeat(tf.expand_dims(u_p, 0), batch_now, axis=0)
        temp_lambda_tensor3 = tf.repeat(tf.expand_dims(lambda_p, 0), batch_now, axis=0)
        temp_theta_tensor3 = tf.expand_dims(tf.expand_dims(theta_p, 0), 2) * tf.ones(
            (batch_now, b - a + 1, latent_dim, b)
        )

        temp_p_c_z = tf.exp(
            tf.reduce_sum(
                (
                    tf.math.log(temp_theta_tensor3)
                    - 0.5 * tf.math.log(2 * np.pi * temp_lambda_tensor3)
                    - tf.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)
                ),
                axis=2,
            )
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
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
    
            # dynamic batch size for this step
            batch_now = tf.shape(z)[0]
    
            # build stacked Z, z_mean, z_log_var
            Z_set, z_mean_set, z_log_var_set = [], [], []
            for n_centroid in range(a, b + 1):
                Z_temp = tf.tile(tf.expand_dims(z, axis=2), [1, 1, n_centroid])
                Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
                Z_set.append(Z_padded)
    
                zm_temp = tf.tile(tf.expand_dims(z_mean, axis=2), [1, 1, n_centroid])
                zm_padded = tf.pad(zm_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
                z_mean_set.append(zm_padded)
    
                zv_temp = tf.tile(tf.expand_dims(z_log_var, axis=2), [1, 1, n_centroid])
                zv_padded = tf.pad(zv_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=-(1e+10))
                z_log_var_set.append(zv_padded)
    
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

            total_loss = reconstruction_loss + args.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # debug
        '''tf.print("theta_p",self.theta_p)
        tf.print("u_p",self.u_p)
        tf.print("lambda_p",self.lambda_p)'''

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def load_pretrain_weights(dmvae, ae):
    #ae = load_model(args.output_hyp + "ae_sim")
    dmvae.encoder.layers[1].set_weights(ae.layers[1].get_weights())
    dmvae.encoder.layers[2].set_weights(ae.layers[2].get_weights())
    dmvae.encoder.layers[3].set_weights(ae.layers[3].get_weights())
    dmvae.encoder.layers[4].set_weights(ae.layers[4].get_weights())
    dmvae.decoder.layers[-1].set_weights(ae.layers[-1].get_weights())
    dmvae.decoder.layers[-2].set_weights(ae.layers[-2].get_weights())
    dmvae.decoder.layers[-3].set_weights(ae.layers[-3].get_weights())
    dmvae.decoder.layers[-4].set_weights(ae.layers[-4].get_weights())
    sample = sample_output.predict(X, batch_size=batch_size)

    # Perform clustering and update dmvae's parameters accordingly
    for n_centroid in range(a, b + 1):
        g = GMM(n_components=n_centroid, covariance_type='diag')
        g.fit(sample)

        means_reshaped = tf.transpose(tf.convert_to_tensor(g.means_, dtype=tf.float32))
        means_padded = tf.pad(means_reshaped, [[0, 0], [0, b - n_centroid]], constant_values=1e-10)
        u_p[n_centroid - a, :, :].assign(means_padded)

        covariances_reshaped = tf.transpose(tf.convert_to_tensor(g.covariances_, dtype=tf.float32))
        covariances_padded = tf.pad(covariances_reshaped, [[0, 0], [0, b - n_centroid]], constant_values=1)
        lambda_p[n_centroid - a, :, :].assign(covariances_padded)

    print('Pretrain weights loaded!')

    return dmvae


def lr_decay():
    # Manually decay the learning rate for the neural network optimizer
    '''current_lr_nn = adam_nn.learning_rate.numpy()
    current_lr_gmm = adam_gmm.learning_rate.numpy()'''

    current_lr_nn = rmsprop_nn.learning_rate.numpy()
    current_lr_gmm = rmsprop_gmm.learning_rate.numpy()
    # Calculate new learning rates with decay
    new_lr_nn = max(current_lr_nn * args.decay_nn, 1e-7)
    new_lr_gmm = max(current_lr_gmm * args.decay_gmm, 1e-7)

    # Update the optimizers with the new learning rates
    '''adam_nn.learning_rate.assign(new_lr_nn)
    adam_gmm.learning_rate.assign(new_lr_gmm)'''

    rmsprop_nn.learning_rate.assign(new_lr_nn)
    rmsprop_gmm.learning_rate.assign(new_lr_gmm)

    # Print the updated learning rates
    print('lr_nn: %f' % rmsprop_nn.learning_rate.numpy())
    print('lr_gmm: %f' % rmsprop_gmm.learning_rate.numpy())

#######################

class EpochBegin(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % decay_n == 0 and epoch != 0:
            lr_decay()

        # Assuming gamma_output is a model that outputs 'gamma' values
        gamma = gamma_output.predict(X, batch_size=batch_size)
        p_k_z, p_c_z = gamma[0], gamma[1]
        # p_k = gamma_output.predict(X, batch_size=batch_size)

        k = np.argmax(tf.reduce_sum(p_k_z, axis=0), axis=-1)
        pk = tf.reduce_sum(p_k_z, axis=0)  # shape (n_k,): total responsibility per k (sums to N, not 1)
        k_list.append(k)
        posteriorK.append(pk.numpy())
        ########## majority vote ##########
        #k_per_subject = tf.argmax(p_k_z, axis=-1)
        #unique_k, _, count_k = tf.unique_with_counts(k_per_subject)
        #majority_k = unique_k[tf.argmax(count_k)]
        #k_majority.append(majority_k)

        k_order = tf.argsort(tf.reduce_sum(p_k_z, axis=0), direction='DESCENDING')  # order of k
        k_order_list.append(k_order.numpy())
        p_c_label = p_c_z[:, k, :]
        assign_c = np.argmax(p_c_label, axis=1)
        assign.append(assign_c)
        acc = cluster_acc(assign_c, Y)
        accuracy.append(acc[0])

        p_truth_label = p_c_z[:, args.truth_k-a, :]
        assign_truth = np.argmax(p_truth_label, axis=1)
        acc_t = cluster_acc(assign_truth, Y)
        accuracy_t.append(acc_t[0])

        ari.append(ARI(Y, assign_c))
        nmi.append(NMI(Y, assign_c, average_method='arithmetic'))
        ari_t.append(ARI(Y, assign_truth))
        nmi_t.append(NMI(Y, assign_truth, average_method='arithmetic'))

        if epoch > 0:
            # print('k:%d' % k)
            print('k_order:', k_order)
            print('acc:%0.8f' % acc[0])
            print("pk", pk)
        
        for cl in range(0, b - a + 1):
            k_value = cl+a
            label = p_c_z[:, cl, :]
            assign_all[k_value] = np.argmax(label, axis=1)
            acc_all[k_value] = cluster_acc(assign_all[k_value], Y)[0]

        for k_value in range(a, b + 1):
            ari_all[k_value] = ARI(Y, assign_all[k_value])
            nmi_all[k_value] = NMI(Y, assign_all[k_value], average_method='arithmetic')


    def on_epoch_end(self, epoch, logs=None):
        for i in range(b-a):
            #weighted pi
            #dmvae.theta_p[i, 0:i+a].assign(dmvae.theta_p[i, 0:i+a] / tf.reduce_sum(dmvae.theta_p[i, 0:i+a]))
            #repadding
            dmvae.theta_p[i, i+a:b].assign(tf.constant(1e-10, shape = [b - (i + a)], dtype=tf.float32))
            dmvae.u_p[i, :, i+a:b].assign(tf.constant(1e-10, shape = [latent_dim,  b - (i+a)], dtype=tf.float32))
            dmvae.lambda_p[i, :, i+a:b].assign(tf.constant(1, shape = [latent_dim,  b - (i+a)], dtype=tf.float32))
       # dmvae.theta_p[13, :].assign(dmvae.theta_p[13, :] / tf.reduce_sum(dmvae.theta_p[13, :]))



best_loss = float('inf')
best_latent_representation = None
best_embedding = None
best_loss_curve = None
best_acc = None
best_assign_c = None
best_k = None
best_pk = None
all_loss = []
all_accuracy = []
all_k = []
all_accuracy_t = []
all_nmi = []
all_ari = []
all_nmi_t = []
all_ari_t = []
all_pk = []

import time as _time_module
_start_time = _time_module.time()
for j in range(0, args.m):
    print(f"Processing iteration {j}...")
    K.clear_session() # Clear previous state
    gc.collect()   # Garbage collection to release unused objects
    ####### AE model setup #######
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    latent = Dense(latent_dim, activation='relu')(h)
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)
    encoder_ae = Model(x, latent, name="encoder")
    AE = Model(x, x_decoded_mean)

    # Compile AE model
    #rmsprop = RMSprop(learning_rate=args.ae_lr, clipnorm=5)
    #AE.compile(optimizer=rmsprop, loss='mean_squared_error')
    ADAM = Adam(learning_rate=args.ae_lr, epsilon=1e-4)
    AE.compile(optimizer=ADAM, loss='binary_crossentropy')

    # Train AE
    fitting_ae = AE.fit(X, X, epochs=args.ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X))


    ae_zmean = encoder_ae.predict(X)
    loss_ae = fitting_ae.history['loss']

    ###### DMVAE ######
    k_list = []
    k_order_list = []
    accuracy = []
    accuracy_t = []
    assign = []
    assign_all = {}
    acc_all = {}
    ari, nmi, ari_t, nmi_t = [], [], [], []
    ari_all, nmi_all = {}, {}
    posteriorK = []

    # Initialize GMM parameters
    theta_p, u_p, lambda_p = gmmpara_init()

    # dmvae model setup
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    # latent_inputs = Input(shape=(latent_dim,))
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)

    # Gamma outputs
    p_k_z, p_c_z = get_gamma()(z)

    # Define models
    sample_output = Model(inputs=x, outputs=z_mean)
    gamma_output = Model(inputs=x, outputs=[p_k_z, p_c_z])
    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(inputs=z, outputs=x_decoded_mean, name="decoder")
    dmvae = DMVAE(encoder, decoder)

    # Load pre-trained weights if applicable
    if ispretrain:
        dmvae = load_pretrain_weights(dmvae, AE)

    # Compile dmvae model
    rmsprop_nn = RMSprop(learning_rate=args.lr_nn, clipnorm=5)
    rmsprop_gmm = RMSprop(learning_rate=args.lr_gmm, clipnorm=5)
    dmvae.compile(optimizer=rmsprop_nn)

    # Train dmvae
    fitting = dmvae.fit(X, shuffle=True, epochs=args.epochs, batch_size=batch_size, callbacks=[EpochBegin()])
    z_mean, _, _ = dmvae.encoder.predict(X, batch_size=batch_size)
    
    # Recompute cluster assignments for the final model
    p_k_z_final, p_c_z_final = gamma_output.predict(X, batch_size=batch_size)
    k_final = np.argmax(np.sum(p_k_z_final, axis=0), axis=-1)
    assign_final = np.argmax(p_c_z_final[:, k_final, :], axis=1)
    assign_truth = np.argmax(p_c_z_final[:, args.truth_k - a, :], axis=1)

    # Save dmvae results
    last_loss = fitting.history['loss'][-1]
    last_recon_loss = fitting.history['reconstruction_loss'][-1]
    last_kl_loss = fitting.history['kl_loss'][-1]

    # Create a combined loss entry
    loss_entry = {
        'loss': last_loss,
        'reconstruction_loss': last_recon_loss,
        'kl_loss': last_kl_loss
    }

    nmi_iter = nmi[-1] if len(nmi) > 0 else 0.0
    ari_iter = ari[-1] if len(ari) > 0 else 0.0
    nmi_iter_t = nmi_t[-1] if len(nmi_t) > 0 else 0.0
    ari_iter_t = ari_t[-1] if len(ari_t) > 0 else 0.0
    posteriorK_arr = np.array(posteriorK)
    all_pk.append(posteriorK_arr[-1])
    
    # Append to all_loss list
    all_loss.append(loss_entry)
    all_accuracy.append(accuracy[-1])
    all_accuracy_t.append(accuracy_t[-1])
    all_k.append(k_list[-1])

    all_nmi.append(nmi_iter)
    all_ari.append(ari_iter)
    all_nmi_t.append(nmi_iter_t)
    all_ari_t.append(ari_iter_t)
    
    if last_loss < best_loss:
        best_loss = last_loss
        best_z_mean = z_mean
        best_loss_curve = fitting.history['loss']
        recon_loss = fitting.history['reconstruction_loss']
        kl_loss = fitting.history['kl_loss']
        best_acc = accuracy
        best_assign_c = assign[-1]
        best_k = k_list
        best_loss_ae = loss_ae
        best_acc_t = accuracy_t
        best_assign_all = assign_all
        best_acc_all = acc_all
        best_ari = ari
        best_nmi = nmi
        best_ari_t = ari_t
        best_nmi_t = nmi_t
        best_ari_all = ari_all.copy()
        best_nmi_all = nmi_all.copy()
        best_pk = posteriorK_arr[-1]
        nmi_best = nmi[-1] if len(nmi) > 0 else 0.0
        ari_best = ari[-1] if len(ari) > 0 else 0.0
        assign_truth_best = best_assign_all[args.truth_k]
        nmi_truth = nmi_t[-1] if len(nmi_t) > 0 else 0.0
        ari_truth = ari_t[-1] if len(ari_t) > 0 else 0.0


        # UMAP visualization for dmvae latent space (only for the best iteration)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        best_embedding = reducer.fit_transform(z_mean)
        best_ae_embedding = reducer.fit_transform(ae_zmean)
        
        save_path = os.path.join(args.output_hyp, "ae_sim")
        AE.save(save_path)
    print(f"Finished iteration {j}...")

    print(f"Finished iteration {j}...")

out = args.output_datafile.rstrip(os.sep) + os.sep
os.makedirs(out.rstrip(os.sep), exist_ok=True)

np.savetxt(os.path.join(out, 'z_mean.txt'), best_z_mean)
np.savetxt(os.path.join(out, 'DMVAE_loss.txt'), best_loss_curve)
np.savetxt(os.path.join(out, 'accuracy.txt'), best_acc)
np.savetxt(os.path.join(out, 'accuracy_t.txt'), best_acc_t)
np.savetxt(os.path.join(out, 'k.txt'), best_k)
np.savetxt(os.path.join(out, 'assign_c.txt'), best_assign_c)
if best_pk is not None:
    # best_pk is total responsibility per k (sums to N), not a probability; divide by N to get p(k)
    n_samples = best_z_mean.shape[0]
    pk_normalized = best_pk / np.maximum(n_samples, 1)
    np.savetxt(os.path.join(out, 'posteriorK_best.txt'), np.atleast_2d(best_pk))
    np.savetxt(os.path.join(out, 'posteriorK_best_normalized.txt'), np.atleast_2d(pk_normalized))
if all_pk:
    np.save(os.path.join(out, 'posteriorK_all_iters.npy'), np.stack(all_pk))
with open(os.path.join(out, 'accuracies_all_k.json'), 'w') as f:
    json.dump({str(k): float(v) for k, v in best_acc_all.items()}, f, indent=2)
with open(os.path.join(out, 'assignments_all_k.json'), 'w') as f:
    json.dump({str(k): np.asarray(best_assign_all[k]).astype(int).tolist() for k in best_assign_all}, f, indent=2)
np.savetxt(os.path.join(out, 'all_accuracy.txt'), np.array(all_accuracy))
np.savetxt(os.path.join(out, 'all_k.txt'), np.array(all_k))
np.savetxt(os.path.join(out, 'all_accuracy_t.txt'), np.array(all_accuracy_t))
np.savetxt(os.path.join(out, 'ari.txt'), np.array(best_ari))
np.savetxt(os.path.join(out, 'nmi.txt'), np.array(best_nmi))
np.savetxt(os.path.join(out, 'ari_t.txt'), np.array(best_ari_t))
np.savetxt(os.path.join(out, 'nmi_t.txt'), np.array(best_nmi_t))
with open(os.path.join(out, 'ari_all.json'), 'w') as f:
    json.dump({int(k): float(v) for k, v in best_ari_all.items()}, f, indent=4)
with open(os.path.join(out, 'nmi_all.json'), 'w') as f:
    json.dump({int(k): float(v) for k, v in best_nmi_all.items()}, f, indent=4)
np.savetxt(os.path.join(out, 'all_ari.txt'), np.array(all_ari))
np.savetxt(os.path.join(out, 'all_nmi.txt'), np.array(all_nmi))
np.savetxt(os.path.join(out, 'all_ari_t.txt'), np.array(all_ari_t))
np.savetxt(os.path.join(out, 'all_nmi_t.txt'), np.array(all_nmi_t))
_end_time = _time_module.time()
np.savez(os.path.join(out, 'dmvae.npz'), ARI=np.array(best_ari), NMI=np.array(best_nmi), K=np.array(best_k), ACC=np.array(best_acc), Embedding=np.array(best_z_mean),
         Clusters=np.array(best_assign_c), Time_use=_end_time - _start_time)


# Save UMAP visualization for the best dmvae latent space (style match dmvae_keras2)
xy = np.column_stack([best_embedding[:, 0], best_embedding[:, 1]])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
cvec_p, cmap_p, uniq_p, cols_p = _get_colors_cmap(best_assign_c)
ax1.scatter(xy[:, 0], xy[:, 1], c=cvec_p, s=1, cmap=cmap_p, vmin=-0.5, vmax=len(uniq_p) - 0.5)
_add_labels(ax1, xy, best_assign_c, {u: cols_p[i] for i, u in enumerate(uniq_p)})
ax1.set_title('Predicted label', fontsize=24)
ax1.legend(handles=[Patch(facecolor=cols_p[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_p)],
           loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
lines = [f'ACC: {best_acc[-1]:.3f}', f'ACC (truth k): {best_acc_t[-1]:.3f}', f'k: {best_k[-1] + a}']
ax1.text(0.98, 0.02, '\n'.join(lines), transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
cvec_t, cmap_t, uniq_t, cols_t = _get_colors_cmap(Y)
ax2.scatter(xy[:, 0], xy[:, 1], c=cvec_t, s=1, cmap=cmap_t, vmin=-0.5, vmax=len(uniq_t) - 0.5)
_add_labels(ax2, xy, Y, {u: cols_t[i] for i, u in enumerate(uniq_t)})
ax2.set_title('True Label', fontsize=24)
ax2.legend(handles=[Patch(facecolor=cols_t[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_t)],
           loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(out, "umap_DMVAE_best.png"), bbox_inches='tight')
plt.close(fig)
plt.clf()

for k in sorted(best_assign_all.keys()):
    pred_k = best_assign_all[k]
    acc_k = best_acc_all.get(k, None)
    ari_k = best_ari_all.get(k, None)
    nmi_k = best_nmi_all.get(k, None)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    xy = np.column_stack([best_embedding[:, 0], best_embedding[:, 1]])
    cvec_p, cmap_p, uniq_p, cols_p = _get_colors_cmap(pred_k)
    ax1.scatter(xy[:, 0], xy[:, 1], c=cvec_p, s=1, cmap=cmap_p, vmin=-0.5, vmax=len(uniq_p) - 0.5)
    _add_labels(ax1, xy, pred_k, {u: cols_p[i] for i, u in enumerate(uniq_p)})
    ax1.set_title(f'Predicted Label (k={k})', fontsize=24)
    ax1.legend(handles=[Patch(facecolor=cols_p[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_p)],
               loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
    lines = []
    if acc_k is not None:
        lines.append(f'ACC: {acc_k:.3f}')
    if ari_k is not None:
        lines.append(f'ARI: {ari_k:.3f}')
    if nmi_k is not None:
        lines.append(f'NMI: {nmi_k:.3f}')
    if lines:
        ax1.text(0.98, 0.02, '\n'.join(lines), transform=ax1.transAxes, ha='right', va='bottom',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
    cvec_t, cmap_t, uniq_t, cols_t = _get_colors_cmap(Y)
    ax2.scatter(xy[:, 0], xy[:, 1], c=cvec_t, s=1, cmap=cmap_t, vmin=-0.5, vmax=len(uniq_t) - 0.5)
    _add_labels(ax2, xy, Y, {u: cols_t[i] for i, u in enumerate(uniq_t)})
    ax2.set_title('True Label', fontsize=24)
    ax2.legend(handles=[Patch(facecolor=cols_t[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_t)],
               loc='center left', bbox_to_anchor=(1, 0.5), title='Classes', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"umap_DMVAE{k}.png"), bbox_inches='tight', dpi=150)
    plt.close(fig)
    plt.clf()

# plot ae UMAP (style match dmvae_keras2)
fig_ae, ax_ae = plt.subplots(1, 1, figsize=(8, 6))
cvec_ae, cmap_ae, uniq_ae, cols_ae = _get_colors_cmap(Y)
ax_ae.scatter(best_ae_embedding[:, 0], best_ae_embedding[:, 1], c=cvec_ae, s=1, cmap=cmap_ae, vmin=-0.5, vmax=len(uniq_ae) - 0.5)
_add_labels(ax_ae, best_ae_embedding, Y, {u: cols_ae[i] for i, u in enumerate(uniq_ae)})
ax_ae.set_title("AE Latent Representation", fontsize=16)
ax_ae.legend(handles=[Patch(facecolor=cols_ae[i], edgecolor='gray', label=str(u)) for i, u in enumerate(uniq_ae)],
             loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
plt.tight_layout()
plt.savefig(os.path.join(out, "ae.png"), bbox_inches='tight')
plt.close(fig_ae)
plt.clf()

# plot ae loss curve
plt.figure(figsize=(6, 4))
plt.plot(best_loss_ae, label='AE loss')
plt.legend()
plt.title('AE Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(os.path.join(out, "ae_loss.png"))
plt.close()
plt.clf()

# Plot DMVAE loss curve
plt.figure(figsize=(8, 6))
plt.plot(best_loss_curve, label='Total Loss')
plt.plot(recon_loss, label='Reconstruction Loss')
plt.plot(kl_loss, label='KL Loss')
plt.legend()
plt.title('DMVAE Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(os.path.join(out, 'DMVAE_loss.png'))
plt.close()
plt.clf()

with open(os.path.join(out, 'all_loss.json'), 'w') as f:
    json.dump(all_loss, f, indent=4)
