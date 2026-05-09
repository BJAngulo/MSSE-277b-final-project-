"""
gan_augment.py

WGAN-GP augmentation module.

Designed for:
- EDF loading via metadata
- fold-safe training
- GroupKFold CV integration
"""

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


# ==========================
# Config
# ==========================
NOISE_DIM = 32
N_CLASSES = 2


# =========================================================
# GENERATOR
# =========================================================
class Generator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + N_CLASSES, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, feature_dim)
        )

    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        return self.net(x)


# ============================================================
# Critic
# ============================================================
class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_dim + N_CLASSES, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)
    

# ============================================================
# Gradient penalty
# ============================================================
def gradient_penalty(C, real, fake, labels, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    score = C(interpolated, labels)

    grads = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return gp

# ============================================================
# WGAN-GP training
# ============================================================
def train_wgan_gp(X_train, y_train, feature_dim,
                  epochs=500,
                  batch_size=32,
                  lr=1e-4,
                  n_critic=10,):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(feature_dim).to(device)
    C = Critic(feature_dim).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.9))

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

            real_score = C(real, labels)
            fake_score = C(fake, labels)

            gp = gradient_penalty(C, real, fake, labels, device)

            loss_C = -(real_score.mean() - fake_score.mean()) + 10 * gp

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


    return G


# ============================================================
# Synthetic generation
# ============================================================
def generate_synthetic_samples(G, n_samples, class_label):
    device = next(G.parameters()).device

    noise = torch.randn(n_samples, NOISE_DIM, device=device)

    labels = torch.zeros(n_samples, 2, device=device)
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


# =========================================================
# MAIN EXPERIMENT
# =========================================================
def run_gan_experiment(X, y, groups,
                       n_splits=5,
                       epochs=500,
                       pool_size=100):

    results = []
    gkf = GroupKFold(n_splits=n_splits)

    feature_dim = X.shape[1]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --------------------
        # Train GAN
        # --------------------
        G = train_wgan_gp(
            X_train,
            y_train,
            feature_dim=feature_dim,
            epochs=epochs
        )

        # --------------------
        # Generate synthetic data
        # --------------------
        fake_0_all = generate_synthetic_samples(G, pool_size * 2, 0)
        fake_1_all = generate_synthetic_samples(G, pool_size * 2, 1)

        fake_0 = filter_by_centroid(fake_0_all, X_train[y_train == 0], pool_size)
        fake_1 = filter_by_centroid(fake_1_all, X_train[y_train == 1], pool_size)

        X_aug = np.vstack([X_train, fake_0, fake_1])
        y_aug = np.concatenate([
                y_train,
                np.zeros(pool_size),
                np.ones(pool_size)
            ])
        
        perm = np.random.permutation(len(X_aug))
        X_aug = X_aug[perm]
        y_aug = y_aug[perm]

        # --------------------
        # Classifier
        # --------------------
        clf = Pipeline([
            ("svm", SVC(kernel="rbf"))
        ])

        clf.fit(X_aug, y_aug)
        pred = clf.predict(X_test)

        results.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, pred),
            "F1": f1_score(y_test, pred)
        })

    return results