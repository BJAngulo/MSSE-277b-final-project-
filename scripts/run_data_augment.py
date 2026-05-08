"""
run_data_augmentation.py

Main experiment runner for EEG augmentation research.

Runs:
1. Standard augmentation (SVM + handcrafted features)
2. VAE-based augmentation
3. GAN augmentation
4. Baseline SVM

Outputs:
- combined results table
- CSV saved to /results
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import torch

# ==========================
# Path setup
# ==========================

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# add project root to path
sys.path.append(str(PROJECT_ROOT))

from src.standard_augment import (
    X, y, groups, filepaths,
    augmentations,
    run_svm_cv
)

from src.vae_augment import run_vae_experiment
from src.gan_augment import run_gan_experiment


# ==========================
# Reproducibility
# ==========================
np.random.seed(42)
torch.manual_seed(42)


# ==========================
# Standard augmentation run
# ==========================
def run_standard_experiments():
    print("\n==============================")
    print("STANDARD AUGMENTATION")
    print("==============================")

    results = []

    for name, aug_fn in augmentations.items():
        print(f"Running: {name}")

        res = run_svm_cv(
            name=name,
            X=X,
            y=y,
            groups=groups,
            filepaths=filepaths,
            augmenter=aug_fn
        )

        results.append(res)

    return pd.DataFrame(results)


# ==========================
# VAE augmentation run
# ==========================
def run_vae_experiments():
    print("\n==============================")
    print("VAE AUGMENTATION")
    print("==============================")

    return run_vae_experiment(
        X=X,
        y=y,
        groups=groups,
        filepaths=filepaths
    )


# ==========================
# GAN augmentation run
# ==========================
def run_gan_experiments():
    print("\n==============================")
    print("GAN AUGMENTATION")
    print("==============================")

    results = run_gan_experiment(
        X=X,
        y=y,
        groups=groups,
        epochs=1000,
        pool_size=200,
        feature_dim=X.shape[1]
    )

    return pd.DataFrame(results)


# ==========================
# Baseline SVM run
# ==========================
def run_baseline():
    print("\n==============================")
    print("BASELINE (NO AUGMENTATION)")
    print("==============================")

    res = run_svm_cv(
        name="Baseline",
        X=X,
        y=y,
        groups=groups,
        filepaths=filepaths,
        augmenter=None
    )

    return pd.DataFrame([res])


# ==========================
# Main execution
# ==========================
def main():

    # ---- Standard ----
    df_standard = run_standard_experiments()
    df_standard["Method"] = "Standard"

    # ---- VAE ----
    df_vae = run_vae_experiments()
    df_vae["Method"] = "VAE"

    # ---- GAN ----
    df_gan = run_gan_experiments()
    df_gan["Method"] = "GAN"

    # ---- Baseline SVM ----
    df_base = run_baseline()
    df_base["Method"] = "Baseline"

    # ---- Combine ----
    results_df = pd.concat(
        [df_base, df_standard, df_vae, df_gan],
        ignore_index=True
    )

    print("\n==============================")
    print("FINAL COMPARISON")
    print("==============================")

    print(results_df)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    results_df.to_csv(results_dir / "augmentation_results.csv", index=False)

    print(f"\nSaved results to: {results_dir}")


if __name__ == "__main__":
    main()