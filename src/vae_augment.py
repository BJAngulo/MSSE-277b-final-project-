"""
vae_augment.py

VAE-based EEG augmentation module.

Provides:
- train_vae()
- generate_samples()
- run_vae_experiment()
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

from src.config import N_SPLITS
from src.metadata import build_metadata
from src.features import extract_band_power_from_array


# ==========================
# Sampling layer
# ==========================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ==========================
# VAE model
# ==========================
def build_encoder(dim, latent):
    inp = layers.Input(shape=(dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)

    z_mean = layers.Dense(latent)(x)
    z_log_var = layers.Dense(latent)(x)

    return models.Model(inp, [z_mean, z_log_var])


def build_decoder(dim, latent):
    inp = layers.Input(shape=(latent,))
    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(dim)(x)

    return models.Model(inp, out)


def vae_loss(x, xr, mean, log_var):
    recon = tf.reduce_mean(tf.reduce_sum((x - xr) ** 2, axis=1))
    kl = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
    )
    return recon + kl


# ==========================
# Training
# ==========================
def train_vae(X, latent=8, epochs=20):

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype("float32")

    enc = build_encoder(X.shape[1], latent)
    dec = build_decoder(X.shape[1], latent)

    opt = tf.keras.optimizers.Adam()
    ds = tf.data.Dataset.from_tensor_slices(Xs).batch(64)

    for _ in range(epochs):
        for batch in ds:
            with tf.GradientTape() as tape:
                m, lv = enc(batch)
                z = Sampling()([m, lv])
                xr = dec(z)

                loss = vae_loss(batch, xr, m, lv)

            grads = tape.gradient(loss, enc.trainable_weights + dec.trainable_weights)
            opt.apply_gradients(zip(grads, enc.trainable_weights + dec.trainable_weights))

    return enc, dec, scaler


# ==========================
# Sampling
# ==========================
def generate_samples(decoder, n, latent_dim):
    z = np.random.normal(size=(n, latent_dim)).astype("float32")
    return decoder.predict(z, verbose=0)


# ==========================
# Experiment runner
# ==========================
def run_vae_experiment(X, y, groups, filepaths):

    cv = GroupKFold(n_splits=N_SPLITS)

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    acc, f1 = [], []

    for train_idx, test_idx in cv.split(X, y, groups):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        enc, dec, scaler = train_vae(X_train, latent=8, epochs=20)

        # augmentation
        z = np.random.normal(size=(len(X_train), 8)).astype("float32")
        X_aug = dec.predict(z, verbose=0)

        X_train_final = np.vstack([X_train, X_aug])
        y_train_final = np.concatenate([y_train, y_train])

        model = clone(base_model)
        model.fit(X_train_final, y_train_final)

        pred = model.predict(X_test)

        acc.append((pred == y_test).mean())
        f1.append(2 * (acc[-1] * acc[-1]) / (acc[-1] + acc[-1] + 1e-8))

    return pd.DataFrame([{
        "Condition": "VAE",
        "Accuracy": np.mean(acc),
        "F1": np.mean(f1)
    }])