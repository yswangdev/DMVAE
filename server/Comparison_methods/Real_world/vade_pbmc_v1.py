import tensorflow as tf
import numpy as np
import pandas as pd
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
from six.moves import cPickle
import math
from sklearn.mixture import GaussianMixture as GMM
from scipy.optimize import linear_sum_assignment
import os
import gc
import argparse


intermediate_dim = [500, 500, 2000]
batch_size = 100
latent_dim = 10
decay_n, decay_nn, decay_gmm, alpha = 10, 0.8, 0.8, 1
ispretrain = True

parser = argparse.ArgumentParser()
parser.add_argument('--lr_nn', type=float, default=1e-6, help="DMVAE Learning rate")
parser.add_argument("--lr_gmm", type=float, default=1e-6, help="Learning rate for GMM")
parser.add_argument("--epochs", type=int, default=100, help="DMVAE epochs")
parser.add_argument('--ae_lr', type=float, default=1e-5, help="SAE learning rate")
parser.add_argument('--ae_epoch', type=int, default=10, help="SAE epoch")
parser.add_argument("--truth_k", type=int, required=True, help="Ground truth cluster count")
parser.add_argument("--input_datafile", type=str, required=True, help="Path to input data directory")
parser.add_argument("--input_file", type=str, required=True, help="input data")
parser.add_argument("--output_datafile", type=str, required=True, help="Path for output results")
parser.add_argument("--output_hyp", type=str, required=True, help="Path for hyperparameter outputs")
parser.add_argument("--m", type=int, default=100, help="number of pretrain times")
parser.add_argument("--n_centroid", type=int, required=True, help="Number of GMM centroids")
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
    "n_centroid": args.n_centroid
}

# Save to a JSON file
with open(args.output_hyp + "hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=4)
    
    
x_t = np.loadtxt(args.input_datafile + args.input_file)
x_t[np.isnan(x_t)] = 0
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt(args.input_datafile + "pbmc_meta.txt")
Y = Y.astype(int)


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
    theta_init = np.ones(n_centroid) / n_centroid
    u_init = np.zeros((latent_dim, n_centroid))
    lambda_init = np.ones((latent_dim, n_centroid))

    theta_p = tf.Variable(theta_init, trainable=True, dtype=tf.float32, name="pi")
    u_p = tf.Variable(u_init, trainable=True, dtype=tf.float32, name="u")
    lambda_p = tf.Variable(lambda_init, trainable=True, dtype=tf.float32, name="lambda")

    return theta_p, u_p, lambda_p

class get_gamma(Layer):
    def call(self, inputs):
        #temp_Z = tf.transpose(K.repeat(inputs, n_centroid), [0, 2, 1])
        temp_Z = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, n_centroid])
        temp_u_tensor3 = tf.repeat(tf.expand_dims(u_p, 0), batch_size, axis=0)
        temp_lambda_tensor3 = tf.repeat(tf.expand_dims(lambda_p, 0), batch_size, axis=0)
        temp_theta_tensor3 = tf.expand_dims(tf.expand_dims(theta_p, 0), 0) * tf.ones((batch_size, latent_dim, n_centroid))

        temp_p_c_z = tf.exp(
            tf.reduce_sum((tf.math.log(temp_theta_tensor3) - 0.5 * tf.math.log(2 * np.pi * temp_lambda_tensor3) -
                           tf.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)), axis=1)) + 1e-10
        gamma = temp_p_c_z / tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True)
        return gamma


class VADE(Model):
    def __init__(self,encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

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

            Z = tf.transpose(K.repeat(z, n_centroid), [0, 2, 1])
            z_mean_t = tf.transpose(K.repeat(z_mean, n_centroid), [0, 2, 1])
            z_log_var_t = tf.transpose(K.repeat(z_log_var, n_centroid), [0, 2, 1])
            u_tensor3 = tf.repeat(tf.expand_dims(self.u_p, 0), batch_size, axis=0)
            lambda_tensor3 = tf.repeat(tf.expand_dims(self.lambda_p, 0), batch_size, axis=0)
            theta_tensor3 = tf.expand_dims(tf.expand_dims(self.theta_p, 0) ,0) * tf.ones((batch_size, latent_dim, n_centroid))

            p_c_z = K.exp(K.sum((K.log(theta_tensor3) - 0.5 * K.log(2 * math.pi * lambda_tensor3) - \
                             K.square(Z - u_tensor3) / (2 * lambda_tensor3)), axis=1)) + 1e-10

            gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)
            gamma_t = K.repeat(gamma, latent_dim)

            reconstruction_loss = alpha * original_dim * tf.keras.losses.mean_squared_error(data, reconstruction)
            kl_loss = K.sum(0.5*gamma_t*(latent_dim*K.log(math.pi*2)+K.log(lambda_tensor3)+K.exp(z_log_var_t)/lambda_tensor3+K.square(z_mean_t-u_tensor3)/lambda_tensor3),axis=(1,2))\
            -0.5*K.sum(z_log_var+1,axis=-1)\
            -K.sum(K.log(K.repeat_elements(tf.expand_dims(self.theta_p, 0),batch_size,0))*gamma,axis=-1)\
            +K.sum(K.log(gamma)*gamma,axis=-1)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        #tf.print("data", data)
        #tf.print("reconstruction", reconstruction)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def load_pretrain_weights(vade):
    ae = load_model(args.output_hyp + "ae_sim")
    vade.encoder.layers[1].set_weights(ae.layers[1].get_weights())
    vade.encoder.layers[2].set_weights(ae.layers[2].get_weights())
    vade.encoder.layers[3].set_weights(ae.layers[3].get_weights())
    vade.encoder.layers[4].set_weights(ae.layers[4].get_weights())
    vade.decoder.layers[-1].set_weights(ae.layers[-1].get_weights())
    vade.decoder.layers[-2].set_weights(ae.layers[-2].get_weights())
    vade.decoder.layers[-3].set_weights(ae.layers[-3].get_weights())
    vade.decoder.layers[-4].set_weights(ae.layers[-4].get_weights())
    sample = sample_output.predict(X, batch_size=batch_size)

    # Perform clustering and update vade's parameters accordingly
    g = GMM(n_components=n_centroid, covariance_type='diag')
    g.fit(sample)
    u_p.assign(tf.convert_to_tensor(g.means_.T, dtype=tf.float32))
    lambda_p.assign(tf.convert_to_tensor(g.covariances_.T, dtype=tf.float32))
    print('Pretrain weights loaded!')

    return vade

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
        assign_c = np.argmax(gamma, axis=1)
        acc = cluster_acc(np.argmax(gamma, axis=1), Y)
        accuracy.append(acc[0])
        pcz.append(list(np.argmax(gamma, axis=1)))

        if epoch > 0:
            print('acc_p_c_z:%0.8f' % acc[0])


reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
n_centroid = args.n_centroid


all_accuracy = []
all_loss = []
best_loss = float('inf')
best_latent_representation = None
best_embedding = None
best_loss_curve = None
best_acc = None

for j in range(0, args.m):
    print(f"Processing iteration {j}...")
    K.clear_session() # Clear previous state
    gc.collect()   # Garbage collection to release unused objects
    ####### SAE model setup #######
    '''x = Input(shape=(original_dim,))
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
        fitting_sae = SAE.fit(X, X, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X))'''
    inp_ae1 = Input(shape=(original_dim,))
    h1 = Dense(intermediate_dim[0], activation='relu', name='enc1')(inp_ae1)
    rec1 = Dense(original_dim, activation='sigmoid', name='dec1')(h1)
    ae1 = Model(inp_ae1, rec1, name='AE1')
    opt = RMSprop(learning_rate=ae_lr, clipnorm=clip_norm)
    ae1.compile(optimizer=opt, loss='mse')
    ae1.fit(X, X, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X), verbose=0)

    # Get H1
    enc1 = Model(inp_ae1, h1)
    H1 = enc1.predict(X, batch_size=batch_size, verbose=0)

    # ----- AE2: h1 (500) -> Dense(500) -> recon(h1)
    inp_ae2 = Input(shape=(intermediate_dim[0],))
    h2 = Dense(intermediate_dim[1], activation='relu', name='enc2')(inp_ae2)
    rec2 = Dense(intermediate_dim[0], activation='relu', name='dec2')(h2)
    ae2 = Model(inp_ae2, rec2, name='AE2')
    ae2.compile(optimizer=opt, loss='mse')
    ae2.fit(H1, H1, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_split=0.0, verbose=0)

    # Get H2
    enc2 = Model(inp_ae2, h2)
    H2 = enc2.predict(H1, batch_size=batch_size, verbose=0)

    # ----- AE3: h2 (500) -> Dense(2000) -> recon(h2)
    inp_ae3 = Input(shape=(intermediate_dim[1],))
    h3 = Dense(intermediate_dim[2], activation='relu', name='enc3')(inp_ae3)
    rec3 = Dense(intermediate_dim[1], activation='relu', name='dec3')(h3)
    ae3 = Model(inp_ae3, rec3, name='AE3')
    ae3.compile(optimizer=opt, loss='mse')
    ae3.fit(H2, H2, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_split=0.0, verbose=0)

    # Get H3
    enc3 = Model(inp_ae3, h3)
    H3 = enc3.predict(H2, batch_size=batch_size, verbose=0)

    # ----- AE4 (bottleneck): h3 (2000) -> Dense(latent_dim) -> recon(h3)
    inp_ae4 = Input(shape=(intermediate_dim[2],))
    z_pre = Dense(latent_dim, activation='relu', name='enc4')(inp_ae4)
    rec4 = Dense(intermediate_dim[2], activation='relu', name='dec4')(z_pre)
    ae4 = Model(inp_ae4, rec4, name='AE4')
    ae4.compile(optimizer=opt, loss='mse')
    ae4.fit(H3, H3, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_split=0.0, verbose=0)

    # ----- Assemble full SAE with pretrained weights -----
    x = Input(shape=(original_dim,), name='sae_input')

    # encoder path
    h1_full = Dense(intermediate_dim[0], activation='relu', name='enc1_full')(x)
    h2_full = Dense(intermediate_dim[1], activation='relu', name='enc2_full')(h1_full)
    h3_full = Dense(intermediate_dim[2], activation='relu', name='enc3_full')(h2_full)
    latent = Dense(latent_dim, activation='relu', name='enc4_full')(h3_full)

    # decoder path (mirror)
    h3_dec_full = Dense(intermediate_dim[2], activation='relu', name='dec4_full')(latent)
    h2_dec_full = Dense(intermediate_dim[1], activation='relu', name='dec3_full')(h3_dec_full)
    h1_dec_full = Dense(intermediate_dim[0], activation='relu', name='dec2_full')(h2_dec_full)
    x_decoded_mean = Dense(original_dim, activation='sigmoid', name='dec1_full')(h1_dec_full)

    # models
    encoder_sae = Model(x, latent, name="encoder")
    SAE = Model(x, x_decoded_mean, name="SAE_full")

    SAE.get_layer('enc1_full').set_weights(ae1.get_layer('enc1').get_weights())
    SAE.get_layer('enc2_full').set_weights(ae2.get_layer('enc2').get_weights())
    SAE.get_layer('enc3_full').set_weights(ae3.get_layer('enc3').get_weights())
    SAE.get_layer('enc4_full').set_weights(ae4.get_layer('enc4').get_weights())

    # decoders (mirror from each shallow AE's decoder)
    SAE.get_layer('dec4_full').set_weights(ae4.get_layer('dec4').get_weights())  # latent->2000
    SAE.get_layer('dec3_full').set_weights(ae3.get_layer('dec3').get_weights())  # 2000->500
    SAE.get_layer('dec2_full').set_weights(ae2.get_layer('dec2').get_weights())  # 500->500
    SAE.get_layer('dec1_full').set_weights(ae1.get_layer('dec1').get_weights())  # 500->orig

    # ---- end-to-end fine-tuning (optional but recommended) ----
    opt_full = RMSprop(learning_rate=ae_lr, clipnorm=clip_norm)
    SAE.compile(optimizer=opt_full, loss='mse')
    fitting_sae = SAE.fit(X, X, epochs=ae_epoch, batch_size=batch_size, shuffle=True, validation_data=(X, X),
                          verbose=0)
    # Save SAE model
    SAE.save(args.output_hyp + "ae_sim")

    ae_zmean = encoder_sae.predict(X)
    loss_sae = fitting_sae.history['loss']

    ###### DMVAE ######
    theta_p,u_p,lambda_p = gmmpara_init()
    pcz = []
    accuracy=[]
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    #latent_inputs = Input(shape=(latent_dim,))
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)

    # Additional model outputs for Gamma and sample_output
    Gamma = get_gamma()(z)
    sample_output = Model(inputs=x, outputs=z_mean)
    gamma_output = Model(inputs=x, outputs=Gamma)
    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
    decoder = Model(inputs=z, outputs=x_decoded_mean, name="decoder")
    # Final model
    vade = VADE(encoder, decoder)
    if ispretrain:
        vade = load_pretrain_weights(vade)
    
    # Compile model
    rmsprop_nn = RMSprop(learning_rate=args.lr_nn, clipnorm=5)
    rmsprop_gmm = RMSprop(learning_rate=args.lr_gmm, clipnorm=5)
    vade.compile(optimizer=rmsprop_nn)

    # Training the model
    fitting = vade.fit(X, shuffle=True, epochs=args.epochs, batch_size=batch_size, callbacks=[EpochBegin()])

    # Save dmvae results
    last_loss = fitting.history['loss'][-1]
    all_loss.append(last_loss)
    all_accuracy.append(accuracy[-1])


    z_mean, _, _ = vade.encoder.predict(X, batch_size=batch_size)
    if last_loss < best_loss:
        best_loss = last_loss
        best_latent_representation = z_mean
        best_loss_curve = fitting.history['loss']
        best_acc = accuracy
        best_assign_c = pcz[-1]
        best_embedding = reducer.fit_transform(z_mean)
        
    print(f"Finished iteration {j}...")

np.savetxt(args.output_datafile + 'all_accuracy.txt', all_accuracy)
np.savetxt(args.output_datafile + 'all_loss.txt', all_loss)
np.savetxt(args.output_datafile + 'vade_loss.txt', best_loss_curve)


# Save UMAP visualization for the best dmvae latent space
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
scatter1 = ax1.scatter(best_embedding[:, 0], best_embedding[:, 1], c=best_assign_c, s=1, cmap='viridis')
ax1.set_title('predicted label', fontsize=24)
ax1.text(0.95, 0.95, f'Accuracy: {best_acc[-1]:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))

scatter2 = ax2.scatter(best_embedding[:, 0], best_embedding[:, 1], c=Y, s=1)
ax2.set_title('True Label', fontsize=24)

class_labels = ["B", "CD4/CD8", "Mono", "NK"]

# Get unique values from Y (assuming they are 1, 2, 3, 4)
unique_classes = [1, 2, 3, 4]  # Adjust if your Y values differ

# Create legend handles and apply custom labels
handles2, _ = scatter2.legend_elements()

# Add legends with custom labels
ax1.legend(*scatter1.legend_elements(), loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")
ax2.legend(handles2, class_labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Classes")

plt.tight_layout()
plt.savefig(args.output_datafile + "umap_vade_best.png", bbox_inches='tight')
plt.show()
plt.close(fig)
plt.clf()

    
plt.figure(figsize=(8, 6))
plt.plot(best_loss_curve)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(output_datafile + 'vade_loss.png')
plt.clf()
