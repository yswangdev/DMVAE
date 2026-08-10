"""DMVAE network: layers, the model, and the mixture-size prior p(k).

Sampling, GetGamma -- Keras layers.
p_k_dist           -- prior over k.
DMVAE, gmmpara_init -- the model and its GMM parameters.
"""

from __future__ import annotations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from typing import Optional
import math
import numpy as np
import tensorflow as tf


# --- Layers ---


class Sampling(Layer):
    """Sample z ~ N(z_mean, z_log_var)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_now = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_now, tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GetGamma(Layer):
    """Posterior over mixture components given latent z."""

    def __init__(self, theta_p, u_p, lambda_p, a: int, b: int, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.theta_p = theta_p
        self.u_p = u_p
        self.lambda_p = lambda_p
        self.a = a
        self.b = b
        self.latent_dim = latent_dim

    def call(self, inputs):
        temp_z_set = []
        batch_now = tf.shape(inputs)[0]
        for n_centroid in range(self.a, self.b + 1):
            z_temp = tf.tile(tf.expand_dims(inputs, axis=2), [1, 1, n_centroid])
            temp_z_padded = tf.pad(
                z_temp, [[0, 0], [0, 0], [0, self.b - n_centroid]], "CONSTANT", constant_values=1e-10
            )
            temp_z_set.append(temp_z_padded)
        temp_z = tf.cast(tf.stack(temp_z_set, axis=1), tf.float32)
        temp_u_tensor3 = tf.repeat(tf.expand_dims(self.u_p, 0), batch_now, axis=0)
        temp_lambda_tensor3 = tf.repeat(tf.expand_dims(self.lambda_p, 0), batch_now, axis=0)
        temp_theta_tensor3 = tf.expand_dims(tf.expand_dims(self.theta_p, 0), 2) * tf.ones(
            (batch_now, self.b - self.a + 1, self.latent_dim, self.b)
        )
        temp_p_c_z = tf.exp(
            tf.reduce_sum(
                (
                    tf.math.log(temp_theta_tensor3)
                    - 0.5 * tf.math.log(2 * np.float32(np.pi) * temp_lambda_tensor3)
                    - tf.square(temp_z - temp_u_tensor3) / (2 * temp_lambda_tensor3)
                ),
                axis=2,
            )
        ) + 1e-10
        gamma = temp_p_c_z / tf.reduce_sum(tf.reduce_sum(temp_p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
        p_k_z = tf.reduce_sum(gamma, axis=-1)
        p_c_z = gamma / tf.reduce_sum(gamma, axis=-1, keepdims=True)
        return p_k_z, p_c_z


# --- Mixture-size prior p(k) ---


def p_k_dist(prior_dist: str, a: int, b: int):
    if prior_dist == "uniform":
        p_k = 1 / (b - a + 1)
        return p_k
    if prior_dist == "poisson":
        k_values = np.arange(a, b + 1)
        poisson_pmf = np.exp(-10) * np.power(10, k_values) / np.array(
            [math.factorial(int(k)) for k in k_values]
        )
        normalization_factor = np.sum(poisson_pmf)
        p_k = poisson_pmf / normalization_factor
        return np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones((100, b - a + 1, b))
    if prior_dist == "geometric":
        k_values = np.arange(a, b + 1)
        geometric_pmf = np.power(0.5, k_values) / 0.5
        normalization_factor = np.sum(geometric_pmf)
        p_k = geometric_pmf / normalization_factor
        return np.expand_dims(np.expand_dims(p_k, 0), 2) * np.ones((100, b - a + 1, b))
    raise ValueError(f"Unknown priorDist {prior_dist!r}")


# --- DMVAE model ---


def gmmpara_init(a: int, b: int, latent_dim: int):
    theta_init = []
    u_init = []
    lambda_init = []
    for n_centroid in range(a, b + 1):
        theta_init.append(np.ones(n_centroid) / n_centroid)
        u_init.append(np.zeros((latent_dim, n_centroid)))
        lambda_init.append(np.ones((latent_dim, n_centroid)))

    theta_init_padded = np.array(
        [np.pad(theta, (0, b - len(theta)), "constant", constant_values=1e-10) for theta in theta_init]
    )
    u_init_padded = np.array(
        [np.pad(u, ((0, 0), (0, b - u.shape[1])), "constant", constant_values=1e-10) for u in u_init]
    )
    lambda_init_padded = np.array(
        [
            np.pad(la, ((0, 0), (0, b - la.shape[1])), "constant", constant_values=1)
            for la in lambda_init
        ]
    )

    theta_p = tf.Variable(theta_init_padded, trainable=True, dtype=tf.float32, name="pi")
    u_p = tf.Variable(u_init_padded, trainable=True, dtype=tf.float32, name="u")
    lambda_p = tf.Variable(lambda_init_padded, trainable=True, dtype=tf.float32, name="lambda")
    return theta_p, u_p, lambda_p


class DMVAE(Model):
    def __init__(
        self,
        encoder,
        decoder,
        theta_p,
        u_p,
        lambda_p,
        *,
        beta: float,
        original_dim: int,
        latent_dim: int,
        a: int,
        b: int,
        alpha: float,
        p_k,
        kl_use_repeat_elements: bool = False,
        fixed_batch_for_kl: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        if np.isnan(theta_p).any() or np.isnan(u_p).any() or np.isnan(lambda_p).any():
            raise ValueError("Initial parameters contain NaNs")

        self.theta_p = theta_p
        self.u_p = u_p
        self.lambda_p = lambda_p
        self.beta = float(beta)
        self.original_dim = int(original_dim)
        self.latent_dim = int(latent_dim)
        self.a = int(a)
        self.b = int(b)
        self.alpha = float(alpha)
        self._p_k = tf.constant(np.asarray(p_k, dtype=np.float32))
        self.kl_use_repeat_elements = bool(kl_use_repeat_elements)
        self.fixed_batch_for_kl = int(fixed_batch_for_kl) if fixed_batch_for_kl is not None else None

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

            z_set = []
            z_mean_set = []
            z_log_var_set = []
            for n_centroid in range(self.a, self.b + 1):
                z_temp = tf.tile(tf.expand_dims(z, axis=2), [1, 1, n_centroid])
                z_padded = tf.pad(z_temp, [[0, 0], [0, 0], [0, self.b - n_centroid]], "CONSTANT", constant_values=1e-10)
                z_set.append(z_padded)
                z_mean_temp = tf.tile(tf.expand_dims(z_mean, axis=2), [1, 1, n_centroid])
                z_mean_padded = tf.pad(
                    z_mean_temp, [[0, 0], [0, 0], [0, self.b - n_centroid]], "CONSTANT", constant_values=1e-10
                )
                z_mean_set.append(z_mean_padded)
                z_log_var_temp = tf.tile(tf.expand_dims(z_log_var, axis=2), [1, 1, n_centroid])
                z_log_var_padded = tf.pad(
                    z_log_var_temp,
                    [[0, 0], [0, 0], [0, self.b - n_centroid]],
                    "CONSTANT",
                    constant_values=-(1e10),
                )
                z_log_var_set.append(z_log_var_padded)
            z_stacked = tf.cast(tf.stack(z_set, axis=1), tf.float32)
            z_mean_t = tf.cast(tf.stack(z_mean_set, axis=1), tf.float32)
            z_log_var_t = tf.cast(tf.stack(z_log_var_set, axis=1), tf.float32)
            u_tensor3 = tf.repeat(tf.expand_dims(self.u_p, 0), batch_now, axis=0)
            lambda_tensor3 = tf.repeat(tf.expand_dims(self.lambda_p, 0), batch_now, axis=0)

            theta_tensor3 = tf.expand_dims(tf.expand_dims(self.theta_p, 0), 2) * tf.ones(
                (batch_now, self.b - self.a + 1, self.latent_dim, self.b)
            )

            p_c_z = (
                K.exp(
                    K.sum(
                        (
                            K.log(theta_tensor3)
                            - 0.5 * K.log(2 * math.pi * lambda_tensor3)
                            - K.square(z_stacked - u_tensor3) / (2 * lambda_tensor3)
                        ),
                        axis=2,
                    )
                )
                + 1e-10
            )

            gamma = p_c_z / tf.reduce_sum(tf.reduce_sum(p_c_z, axis=-1, keepdims=True), axis=1, keepdims=True)
            gamma_t = tf.repeat(tf.expand_dims(gamma, 2), self.latent_dim, axis=2)

            reconstruction_loss = self.alpha * self.original_dim * tf.keras.losses.mean_squared_error(
                data, reconstruction
            )
            k13 = K.exp(z_log_var_t) / lambda_tensor3
            if self.kl_use_repeat_elements:
                bs = self.fixed_batch_for_kl
                mix = K.repeat_elements(tf.expand_dims(self.theta_p, 0), bs, 0) * tf.cast(self._p_k, tf.float32)
                theta_pk_log = K.log(mix)
            else:
                mix = (
                    tf.repeat(tf.expand_dims(self.theta_p, 0), repeats=batch_now, axis=0)
                    * tf.expand_dims(self._p_k, 0)
                    + 1e-10
                )
                theta_pk_log = K.log(mix)
            kl_loss = (
                K.sum(
                    0.5
                    * gamma_t
                    * (
                        self.latent_dim * K.log(math.pi * 2)
                        + K.log(lambda_tensor3)
                        + k13
                        + K.square(z_mean_t - u_tensor3) / lambda_tensor3
                    ),
                    axis=(1, 2, 3),
                )
                - 0.5 * K.sum(z_log_var + 1, axis=-1)
                - K.sum(K.sum(theta_pk_log * gamma, axis=-1), axis=1)
                + K.sum(K.sum(K.log(gamma) * gamma, axis=-1), axis=1)
            )
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        grads_and_vars = [
            (tf.clip_by_norm(grad, 1.0), variable)
            for grad, variable in zip(grads, self.trainable_weights)
            if grad is not None
        ]
        self.optimizer.apply_gradients(grads_and_vars)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
