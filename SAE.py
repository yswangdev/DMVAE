import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import RMSprop, Adam

#intermediate_dim = [1024, 512, 256]
#intermediate_dim = [500, 500, 2000]
intermediate_dim = [250, 250, 1000]
batch_size = 100
latent_dim = 10

#load data
'''input_datafile='/Volumes/SSD/MCW/Research/Data/Single cell simulation/simscaleselect.txt'
x_t = np.loadtxt(input_datafile)
x_t[np.isnan(x_t)] = 0
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt("/Volumes/SSD/MCW/Research/Data/Single cell simulation/simmeta.txt")
Y = Y.astype(int)
'''
from sklearn.datasets import make_blobs

# Generate a synthetic dataset
'''X, Y = make_blobs(n_samples=300, centers=4, cluster_std=0.10, random_state=0)
original_dim = X.shape[1]

np.savetxt("/Volumes/SSD/MCW/Research/Aim 1/VaDE/Simulation data/make_blobs/x.txt", X)
np.savetxt("/Volumes/SSD/MCW/Research/Aim 1/VaDE/Simulation data/make_blobs/y.txt", Y)'''

#laod single-cell simulation data
'''input_datafile='/Volumes/SSD/MCW/Research/Aim 1/sim_data/'
x_t = np.loadtxt(input_datafile + "simscaleselect/simscaleselect_2clus_5.txt")
x_t[np.isnan(x_t)] = 0
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt(input_datafile + "simmeta/simmeta_2clus_5.txt")
Y = Y.astype(int)'''

input_datafile='/Volumes/SSD/MCW/Research/Codes/Simulation_single_cell/scenario3_n10000/'
x_t = np.loadtxt(input_datafile + "simscaleselect3_1.txt")
x_t[np.isnan(x_t)] = 0
X = x_t
original_dim = x_t.shape[1]
Y = np.loadtxt(input_datafile + "simmeta3_1.txt")
Y = Y.astype(int)

# SAE
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim[0], activation='relu')(x)
h = Dense(intermediate_dim[1], activation='relu')(h)
h = Dense(intermediate_dim[2], activation='relu')(h)
latent = Dense(latent_dim, activation='relu')(h)
h_decoded = Dense(intermediate_dim[-1], activation='relu')(latent)
h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
x_decoded_mean = Dense(original_dim, activation="sigmoid")(h_decoded)
encoder = Model(x, latent, name="encoder")
SAE = Model(x, x_decoded_mean)

learning_rate = 1e-5
clip_norm = 5
rmsprop = RMSprop(learning_rate=learning_rate, clipnorm=clip_norm)
SAE.compile(optimizer=rmsprop, loss='mean_squared_error')
fitting = SAE.fit(X, X, epochs=5, batch_size=batch_size, shuffle=True, validation_data=(X, X))

'''learning_rate = 1e-5
adam_nn = Adam(learning_rate=learning_rate, epsilon=1e-4)
SAE.compile(optimizer=adam_nn, loss='mean_squared_error')
SAE.fit(X, X, epochs=20, batch_size=batch_size, shuffle=True, validation_data=(X, X))'''

import matplotlib.pyplot as plt

#output_datafile = '/Users/enid/PycharmProjects/VADE/'
#SAE.save(output_datafile + "ae_sim")

# Assuming the latent space is 2D
latent_representation = encoder.predict(X)
#np.savetxt("/Volumes/SSD/MCW/Research/Aim 1/VaDE/results/09252024/2_clusters/ae_zmean.txt", latent_representation)
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(latent_representation)
plt.scatter(embedding[:, 0], embedding[:, 1], c=Y, s=1)  # 'labels' is your data labels if available
plt.colorbar()
plt.title("Latent Space Representation")
#plt.savefig("/Volumes/SSD/MCW/Research/Aim 1/VaDE/results/09252024/2_clusters/ae.png")
plt.show()

'''loss = fitting.history['loss']
np.savetxt('/Volumes/SSD/MCW/Research/Aim 1/VaDE/results/09252024/2_clusters/ae_loss.txt', loss)'''
