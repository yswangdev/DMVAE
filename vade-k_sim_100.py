import tensorflow as tf
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import RMSprop, Adam
import umap
import sklearn.metrics
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import model_from_json
from tensorflow.keras.callbacks import Callback
import gzip
from six.moves import cPickle
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math
from sklearn.mixture import GaussianMixture as GMM
import os
import gc

#intermediate_dim = [1024, 512, 256]
#intermediate_dim = [500, 500, 2000]
intermediate_dim = [250, 250, 1000]
batch_size = 100
latent_dim = 10
lr_nn, lr_gmm, decay_n, decay_nn, decay_gmm, alpha, epochs = 1e-6, 1e-6, 10, 0.8, 0.8, 1, 50
ispretrain = True
ae_lr = 5e-6
ae_epoch = 10
truth_k = 3
beta = 5

# Paths
input_datafile = '/Volumes/SSD/MCW/Research/Codes/Simulation_single_cell/scenario3_n10000_closer/'
output_base_path = '/Volumes/SSD/MCW/Research/Aim 1/VaDE/results/02242025/r2/sim'

start = 1
end = 2
m=1

import json

# Example hyperparameters
hyperparams = {
    "scenario": 3,
    "batch_size": batch_size,
    "ae epochs": ae_epoch,
    "ae learning rate": ae_lr,
    "epochs": epochs,
    "learning_rate": lr_nn,
    "latent_dim": latent_dim,
    "#pretrain": m,
    "optimizer": "RMSprop",
    "loss_function": "binary_crossentropy",
    "layers": intermediate_dim
}

# Save to a JSON file
with open(output_base_path + "hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=4)

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

a = 2
b = 6
p_k = p_k_dist("uniform")
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


########### cluster accuracy ###########
from scipy.optimize import linear_sum_assignment


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
        temp_Z_set = []
        for n_centroid in range(a, b + 1):
            Z_temp = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, n_centroid])
            temp_Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
            temp_Z_set.append(temp_Z_padded)
        temp_Z = tf.cast(tf.stack(temp_Z_set, axis=1), tf.float32)
        temp_u_tensor3 = tf.repeat(tf.expand_dims(u_p, 0), batch_size, axis=0)
        temp_lambda_tensor3 = tf.repeat(tf.expand_dims(lambda_p, 0), batch_size, axis=0)
        temp_theta_tensor3 = tf.expand_dims(tf.expand_dims(theta_p, 0), 2) * tf.ones(
            (batch_size, b - a + 1, latent_dim, b))

        temp_p_c_z = tf.exp(
            tf.reduce_sum((tf.math.log(temp_theta_tensor3) - 0.5 * tf.math.log(2 * np.pi * temp_lambda_tensor3) -
                           tf.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)), axis=2)) + 1e-10
        gamma = temp_p_c_z / tf.reduce_sum(tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
        p_k_z = tf.reduce_sum(gamma, axis=-1)
        p_c_z = gamma / tf.reduce_sum(gamma, axis=-1, keepdims=True)
        return p_k_z, p_c_z

class dmvae(Model):
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
            # Assuming `z`, `z_mean`, `z_log_var` can be derived from `y_pred` or are accessible as attributes of the class
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            Z_set = []
            z_mean_set = []
            z_log_var_set = []
            for n_centroid in range(a, b + 1):
                Z_temp = tf.tile(tf.expand_dims(z, axis=2), [1, 1, n_centroid])
                Z_padded = tf.pad(Z_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT", constant_values=1e-10)
                Z_set.append(Z_padded)
                z_mean_temp = tf.tile(tf.expand_dims(z_mean, axis=2), [1, 1, n_centroid])
                z_mean_padded = tf.pad(z_mean_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT",
                                       constant_values=1e-10)
                z_mean_set.append(z_mean_padded)
                z_log_var_temp = tf.tile(tf.expand_dims(z_log_var, axis=2), [1, 1, n_centroid])
                z_log_var_padded = tf.pad(z_log_var_temp, [[0, 0], [0, 0], [0, b - n_centroid]], "CONSTANT",
                                          constant_values=-(1e+10))
                z_log_var_set.append(z_log_var_padded)
            Z = tf.cast(tf.stack(Z_set, axis=1), tf.float32)
            z_mean_t = tf.cast(tf.stack(z_mean_set, axis=1), tf.float32)
            z_log_var_t = tf.cast(tf.stack(z_log_var_set, axis=1), tf.float32)
            u_tensor3 = tf.repeat(tf.expand_dims(self.u_p, 0), batch_size, axis=0)
            lambda_tensor3 = tf.repeat(tf.expand_dims(self.lambda_p, 0), batch_size, axis=0)
            # lambda_tensor3_checked = tf.debugging.check_numerics(lambda_tensor3, "lambda_tensor3 has NaN or inf values") #debug
            theta_tensor3 = tf.expand_dims(tf.expand_dims(self.theta_p, 0), 2) * tf.ones(
                (batch_size, b - a + 1, latent_dim, b))

            p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                                 K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=2)) + 1e-10

            gamma = p_c_z / tf.reduce_sum(tf.reduce_sum(p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
            gamma_t = tf.repeat(tf.expand_dims(gamma, 2), latent_dim, axis=2)

            reconstruction_loss = alpha * original_dim * tf.keras.losses.mean_squared_error(data, reconstruction)
            k13 = K.exp(z_log_var_t) / lambda_tensor3
            # k13_n = tf.where(tf.math.is_inf(k13), tf.fill(tf.shape(k13), 1e-5) , k13) #replace inf with 0
            kl_loss = K.sum(0.5 * gamma_t * (latent_dim * K.log(math.pi * 2) + K.log(lambda_tensor3) + k13 + K.square(
                z_mean_t - u_tensor3) / lambda_tensor3), axis=(1, 2, 3)) \
                      - 0.5 * K.sum(z_log_var + 1, axis=-1) \
                      - K.sum(K.sum(
                K.log(K.repeat_elements(tf.expand_dims(self.theta_p, 0), batch_size, 0) * p_k) * gamma,
                axis=-1), axis=1) \
                      + K.sum(K.sum(K.log(gamma) * gamma, axis=-1), axis=1)
            total_loss = reconstruction_loss + beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        '''for grad in grads:
            if tf.reduce_any(tf.math.is_nan(grad)):
                raise ValueError("NaN detected in gradients")'''
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]  # Apply gradient clipping
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

def load_pretrain_weights(dmvae):
    ae = load_model("ae_sim")
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
    new_lr_nn = max(current_lr_nn * decay_nn, 1e-7)
    new_lr_gmm = max(current_lr_gmm * decay_gmm, 1e-7)

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

        ########## sum up ##########
        k = np.argmax(tf.reduce_sum(p_k_z, axis=0), axis=-1)
        pk = tf.reduce_sum(p_k_z, axis=0)
        k_list.append(k)
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

        p_truth_label = p_c_z[:, truth_k-a, :]
        assign_truth = np.argmax(p_truth_label, axis=1)
        acc_t = cluster_acc(assign_truth, Y)
        accuracy_t.append(acc_t[0])
        if epoch > 0:
            #print('k:%d' % k)
            print('k_order:', k_order)
            print('acc:%0.8f' % acc[0])
            print("pk", pk)
            #print("confusion matrix", sklearn.metrics.confusion_matrix(Y, assign_c))
            #print('acc_t:%0.8f' % acc_t[0])
            #print('theta_p:', dmvae.theta_p)
            '''if epoch in {1, 5, 9, 19, 29, 39, 49, 99, 199, 299, 399}:
                np.savetxt(f'{output_datafile}epoch{epoch}_pckz.txt', p_c_z[0])
                np.savetxt(f'{output_datafile}epoch{epoch}_pk.txt', pk)
                z_mean, _, _ = dmvae.encoder.predict(X, batch_size=batch_size)
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
                embedding = reducer.fit_transform(z_mean)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
                scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=assign_c, s=1, cmap='viridis')
                ax1.set_title('predicted label', fontsize=24)
                ax1.legend(*scatter1.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

                scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=Y, s=1)
                ax2.set_title('True Label', fontsize=24)
                ax2.legend(*scatter2.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

                plt.tight_layout()

                # Construct the filename to include the epoch number and save the figure
                filename = f'{output_datafile}UMAP_epoch{epoch}.png'
                plt.savefig(filename, bbox_inches='tight')
                plt.close(fig)

                fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 14))
                scatter3 = ax3.scatter(embedding[:, 0], embedding[:, 1], c=assign_truth, s=1, cmap='viridis')
                ax3.set_title('predicted label', fontsize=24)
                ax3.legend(*scatter3.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

                scatter4 = ax4.scatter(embedding[:, 0], embedding[:, 1], c=Y, s=1)
                ax4.set_title('True Label', fontsize=24)
                ax4.legend(*scatter4.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

                plt.tight_layout()

                # Construct the filename to include the epoch number and save the figure
                filename = f'{output_datafile}UMAP_epoch{epoch}_truth.png'
                plt.savefig(filename, bbox_inches='tight')
                plt.clf()'''


    def on_epoch_end(self, epoch, logs=None):
        for i in range(b-a):
            #weighted pi
            #dmvae.theta_p[i, 0:i+a].assign(dmvae.theta_p[i, 0:i+a] / tf.reduce_sum(dmvae.theta_p[i, 0:i+a]))
            #repadding
            dmvae.theta_p[i, i+a:b].assign(tf.constant(1e-10, shape = [b - (i + a)], dtype=tf.float32))
            dmvae.u_p[i, :, i+a:b].assign(tf.constant(1e-10, shape = [latent_dim,  b - (i+a)], dtype=tf.float32))
            dmvae.lambda_p[i, :, i+a:b].assign(tf.constant(1, shape = [latent_dim,  b - (i+a)], dtype=tf.float32))
       # dmvae.theta_p[13, :].assign(dmvae.theta_p[13, :] / tf.reduce_sum(dmvae.theta_p[13, :]))



import time
start_time = time.time()
# Loop through the files
for i in range(start, end):
    print(f"Processing simulation {i}...")

    # Load data
    x_t = np.loadtxt(input_datafile + f"simscaleselect3_{i}.txt")
    x_t[np.isnan(x_t)] = 0
    X = x_t
    original_dim = x_t.shape[1]
    Y = np.loadtxt(input_datafile + f"simmeta3_{i}.txt")
    Y = Y.astype(int)

    # Define folder to save results
    output_datafile = output_base_path + str(i) + '/'
    if not os.path.exists(output_datafile):
        os.makedirs(output_datafile)

    # Variables to track the lowest loss and corresponding results
    best_loss = float('inf')
    best_latent_representation = None
    best_embedding = None
    best_loss_curve = None
    best_acc = None
    best_assign_c = None
    best_k = None
    all_loss = []
    all_accuracy = []
    all_accuracy_t = []
    all_k = []
    #all_k_majority = []

    for j in range(0, m):
        print(f"Processing simulation {i} iteration {j}...")
        K.clear_session() # Clear previous state
        gc.collect()   # Garbage collection to release unused objects
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

        # Compile SAE model
        rmsprop = RMSprop(learning_rate=ae_lr, clipnorm=5)
        SAE.compile(optimizer=rmsprop, loss='mean_squared_error')

        # Train SAE
        fitting_sae = SAE.fit(X, X, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X))

        # Save SAE model
        SAE.save("/Users/enid/PycharmProjects/dmvae/ae_sim")

        # Get latent space representation
        '''latent_representation = encoder.predict(X)
        np.savetxt(output_datafile + "ae_zmean.txt", latent_representation)

        # UMAP visualization
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        embedding = reducer.fit_transform(latent_representation)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=Y, s=1)
        plt.colorbar()
        plt.title("Latent Space Representation")
        plt.savefig(output_datafile + "ae.png")
        plt.show()

        # Save SAE loss
        loss = fitting.history['loss']
        np.savetxt(output_datafile + 'ae_loss.txt', loss)'''
        ae_zmean = encoder_sae.predict(X)
        loss_sae = fitting_sae.history['loss']

        ###### DMVAE ######
        k_list = []
       # k_majority = []
        k_order_list = []
        accuracy = []
        accuracy_t = []
        assign = []

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
        dmvae = dmvae(encoder, decoder)

        # Load pre-trained weights if applicable
        if ispretrain:
            dmvae = load_pretrain_weights(dmvae)

        # Compile dmvae model
        rmsprop_nn = RMSprop(learning_rate=lr_nn, clipnorm=5)
        rmsprop_gmm = RMSprop(learning_rate=lr_gmm, clipnorm=5)
        dmvae.compile(optimizer=rmsprop_nn)

        # Train dmvae
        fitting = dmvae.fit(X, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[EpochBegin()])

        # save k and accuracy for each model
        #np.savetxt(f'{output_datafile}accuracy_{j}.txt', accuracy)
        #np.savetxt(f'{output_datafile}k_{j}.txt', k_list)

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

        # Append to all_loss list
        all_loss.append(loss_entry)
        all_accuracy.append(accuracy[-1])
        all_accuracy_t.append(accuracy_t[-1])
        all_k.append(k_list[-1])
        #all_k_majority.append(k_majority[-1])
        z_mean, _, _ = dmvae.encoder.predict(X, batch_size=batch_size)
        if last_loss < best_loss:
            best_loss = last_loss
            best_z_mean = z_mean
            best_loss_curve = fitting.history['loss']
            recon_loss = fitting.history['reconstruction_loss']
            kl_loss = fitting.history['kl_loss']
            best_acc = accuracy
            best_assign_c = assign[-1]
            best_k = k_list
            best_loss_sae = loss_sae
            best_acc_t = accuracy_t
            #best_k_majority = k_majority

            # UMAP visualization for dmvae latent space (only for the best iteration)
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
            best_embedding = reducer.fit_transform(z_mean)
            best_sae_embedding = reducer.fit_transform(ae_zmean)
        print(f"Finished simulation {i} iteration {j}...")

    #np.savetxt(output_datafile + 'umap_dmvae-k.txt', best_embedding)
    np.savetxt(output_datafile + 'z_mean.txt', best_z_mean)
    np.savetxt(output_datafile + 'dmvae_loss.txt', best_loss_curve)
    np.savetxt(output_datafile + 'accuracy.txt', best_acc)
    np.savetxt(output_datafile + 'accuracy_t.txt', best_acc_t)
    np.savetxt(output_datafile + 'k.txt', best_k)
    np.savetxt(output_datafile + 'assign_c.txt', best_assign_c)
    #np.savetxt(output_datafile + 'k_majority.txt', best_k_majority)

    # Save UMAP visualization for the best dmvae latent space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    scatter1 = ax1.scatter(best_embedding[:, 0], best_embedding[:, 1], c=best_assign_c, s=1, cmap='viridis')
    ax1.set_title('predicted label', fontsize=24)
    ax1.legend(*scatter1.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

    scatter2 = ax2.scatter(best_embedding[:, 0], best_embedding[:, 1], c=Y, s=1)
    ax2.set_title('True Label', fontsize=24)
    ax2.legend(*scatter2.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
    plt.tight_layout()
    plt.savefig(output_datafile + "umap_dmvae_best.png", bbox_inches='tight')
    plt.close(fig)
    plt.clf()

    # plot sae UMAP
    plt.figure(figsize=(8, 6))
    plt.scatter(best_sae_embedding[:, 0], best_sae_embedding[:, 1], c=Y, s=1)
    plt.colorbar()
    plt.title("SAE Latent Representation")
    plt.savefig(output_datafile + "ae.png")
    plt.clf()

    # plot ae loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(best_loss_sae)
    plt.title('SAE Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(output_datafile + "ae_loss.png")
    plt.clf()

    # Plot loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(best_loss_curve, label = 'Total Loss')
    plt.plot(recon_loss, label = 'Reconstruction Loss')
    plt.plot(kl_loss, label = 'KL Loss')
    plt.legend()
    plt.title('dmvae Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(output_datafile + 'dmvae_loss.png')
    plt.clf()

    print(f"Finished processing simulation {i}.")
    #np.savetxt(output_datafile + 'all_loss.txt', all_loss)
    with open(output_datafile + 'all_loss.json', 'w') as f:
        json.dump(all_loss, f, indent=4)
    np.savetxt(output_datafile + 'all_accuracy.txt', all_accuracy)
    np.savetxt(output_datafile + 'all_k.txt', all_k)
    np.savetxt(output_datafile + 'all_accuracy_t.txt', all_accuracy_t)
    #np.savetxt(output_datafile + 'all_k_majority.txt', all_k_majority)
print("All simulations processed.")
end_time = time.time()
print(f"Total time: {end_time - start_time} seconds.")