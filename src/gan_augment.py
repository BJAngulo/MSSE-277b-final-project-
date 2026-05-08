"""
gan_augment.py

RAW EEG WGAN-GP augmentation module.

Now operates on:
    raw EEG signals (channels x time)

NOT feature space.

Designed for:
- EDF loading via metadata
- fold-safe training
- GroupKFold CV integration
"""

import numpy as np
import torch
import torch.nn as nn
import mne

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


# ==========================
# Config
# ==========================
NOISE_DIM = 32
N_CLASSES = 2


# ============================================================
# Utility: load raw EEG
# ============================================================
def load_raw_eeg(filepath, channels=None):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    if channels is not None:
        raw.pick(channels)

    data = raw.get_data()  # (channels, time)

    return data


def pad_or_crop(signal, target_len):
    """
    Ensure all EEG signals have same time dimension.
    """
    c, t = signal.shape

    if t > target_len:
        return signal[:, :target_len]

    if t < target_len:
        pad = np.zeros((c, target_len - t))
        return np.hstack([signal, pad])

    return signal


# ============================================================
# Generator (raw EEG)
# ============================================================
class Generator(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.feature_shape = feature_shape
        self.out_dim = int(np.prod(feature_shape))

        self.model = nn.Sequential(
            nn.Linear(NOISE_DIM + N_CLASSES, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, self.out_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        x = self.model(x)
        return x.view(x.size(0), *self.feature_shape)


# ============================================================
# Critic
# ============================================================
class Critic(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.in_dim = int(np.prod(feature_shape))

        self.model = nn.Sequential(
            nn.Linear(self.in_dim + N_CLASSES, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1)
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        x = torch.cat([x, labels], dim=1)
        return self.model(x)


# ============================================================
# Gradient penalty
# ============================================================
def compute_gradient_penalty(C, real, fake, labels):
    alpha = torch.rand(real.size(0), 1, device=real.device)

    real_flat = real.view(real.size(0), -1)
    fake_flat = fake.view(fake.size(0), -1)

    interpolated = (alpha * real_flat + (1 - alpha) * fake_flat).requires_grad_(True)

    c_interp = C(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=c_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(c_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# ============================================================
# WGAN-GP training
# ============================================================
def train_wgan_gp(X_train, y_train, feature_shape,
                  epochs=500,
                  batch_size=16,
                  lr=1e-4,
                  lambda_gp=10,
                  n_critic=5,
                  verbose=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(feature_shape).to(device)
    C = Critic(feature_shape).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=lr, betas=(0, 0.9))

    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train.astype(int), device=device)

    y_onehot = torch.zeros(len(y_train), N_CLASSES, device=device)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)

    n = len(X)

    for epoch in range(epochs):

        # ---------------------
        # Train Critic
        # ---------------------
        for _ in range(n_critic):

            idx = torch.randint(0, n, (batch_size,), device=device)

            real = X[idx]
            labels = y_onehot[idx]

            noise = torch.randn(batch_size, NOISE_DIM, device=device)
            fake = G(noise, labels).detach()

            gp = compute_gradient_penalty(C, real, fake, labels)

            loss_C = -C(real, labels).mean() + C(fake, labels).mean() + lambda_gp * gp

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

        # ---------------------
        # Train Generator
        # ---------------------
        idx = torch.randint(0, n, (batch_size,), device=device)
        labels = y_onehot[idx]

        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        fake = G(noise, labels)

        loss_G = -C(fake, labels).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1} | C: {loss_C.item():.4f} | G: {loss_G.item():.4f}")

    return G


# ============================================================
# Synthetic generation
# ============================================================
def generate_synthetic_samples(G, n_samples, class_label, feature_shape):
    device = next(G.parameters()).device

    noise = torch.randn(n_samples, NOISE_DIM, device=device)

    labels = torch.zeros(n_samples, N_CLASSES, device=device)
    labels[:, class_label] = 1

    with torch.no_grad():
        fake = G(noise, labels).cpu().numpy()

    return fake


# ============================================================
# Centroid filtering
# ============================================================
def filter_by_centroid(fake_samples, real_samples, n_keep):
    if len(real_samples) == 0:
        return fake_samples[:n_keep]

    real_flat = real_samples.reshape(real_samples.shape[0], -1)
    fake_flat = fake_samples.reshape(fake_samples.shape[0], -1)

    centroid = real_flat.mean(axis=0)

    fake_norm = np.linalg.norm(fake_flat, axis=1, keepdims=True) + 1e-8
    cent_norm = np.linalg.norm(centroid) + 1e-8

    sims = (fake_flat / fake_norm) @ (centroid / cent_norm)

    top_idx = np.argsort(sims)[::-1][:n_keep]

    return fake_samples[top_idx]


# ============================================================
# MAIN experiment runner
# ============================================================
def run_gan_experiment(X, y, groups,
                       filepaths,
                       metadata_df,
                       channels=None,
                       n_splits=5,
                       epochs=500,
                       pool_size=100):

    results = []
    gkf = GroupKFold(n_splits=n_splits)

    # estimate shape from first sample
    sample = load_raw_eeg(metadata_df.iloc[0]["filepath"], channels)
    feature_shape = sample.shape  # (channels, time)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):

        y_train, y_test = y[train_idx], y[test_idx]

        train_files = metadata_df.iloc[train_idx]["filepath"].values
        test_files  = metadata_df.iloc[test_idx]["filepath"].values

        # ---------------------
        # load raw EEG
        # ---------------------
        X_train = []
        for fp in train_files:
            sig = load_raw_eeg(fp, channels)
            sig = pad_or_crop(sig, feature_shape[1])
            X_train.append(sig)

        X_train = np.stack(X_train)

        X_test = []
        for fp in test_files:
            sig = load_raw_eeg(fp, channels)
            sig = pad_or_crop(sig, feature_shape[1])
            X_test.append(sig)

        X_test = np.stack(X_test)

        # ---------------------
        # scale
        # ---------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
        X_test  = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)

        # ---------------------
        # train GAN
        # ---------------------
        G = train_wgan_gp(X_train, y_train, feature_shape, epochs=epochs)

        # ---------------------
        # generate synthetic
        # ---------------------
        fake_mdd = generate_synthetic_samples(G, pool_size, 1, feature_shape)
        fake_hs  = generate_synthetic_samples(G, pool_size, 0, feature_shape)

        # ---------------------
        # filter
        # ---------------------
        n_mdd = max(10, (y_train == 1).sum() // 3)
        n_hs  = max(10, (y_train == 0).sum() // 3)

        fake_mdd = filter_by_centroid(fake_mdd, X_train[y_train == 1], n_mdd)
        fake_hs  = filter_by_centroid(fake_hs,  X_train[y_train == 0], n_hs)

        # ---------------------
        # flatten for classifier
        # ---------------------
        X_aug = np.vstack([
            X_train.reshape(len(X_train), -1),
            fake_mdd.reshape(len(fake_mdd), -1),
            fake_hs.reshape(len(fake_hs), -1)
        ])

        y_aug = np.concatenate([
            y_train,
            np.ones(n_mdd),
            np.zeros(n_hs)
        ])

        X_test_flat = X_test.reshape(len(X_test), -1)

        # ---------------------
        # classifier
        # ---------------------

        clf = SVC(kernel="rbf")
        clf.fit(X_aug, y_aug)
        pred = clf.predict(X_test_flat)

        results.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, pred),
            "F1": f1_score(y_test, pred)
        })

    return results