"""
standard_augmentation.py

EEG augmentation module:
- Band-power features
- SVM + GroupKFold evaluation
- Time + feature augmentations

Designed for integration with:
    run_data_augmentation.py
"""

import numpy as np
import pandas as pd
import mne

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

from config import N_SPLITS, COMMON_CHANNELS
from scr.metadata import build_metadata
from scr.features import extract_band_power_from_array


# ======================================================
# DATA LOADING
# ======================================================

def build_dataset():
    metadata_df = build_metadata()

    features, labels, groups, filepaths = [], [], [], []

    for _, row in metadata_df.iterrows():

        raw = mne.io.read_raw_edf(row["filepath"], preload=True, verbose=False)
        raw.pick(COMMON_CHANNELS)

        data = raw.get_data()
        sfreq = raw.info["sfreq"]

        feat = extract_band_power_from_array(data, sfreq)

        features.append(feat)
        labels.append(row["label"])
        groups.append(row["patient_id"])
        filepaths.append(row["filepath"])

    return (
        np.vstack(features),
        np.array(labels),
        np.array(groups),
        np.array(filepaths)
    )


# ======================================================
# AUGMENTATIONS
# ======================================================

def load_raw(fp):
    raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
    raw.pick(COMMON_CHANNELS)
    return raw


def time_masking(data, mask_size=900):
    n_ch, n_t = data.shape
    start = np.random.randint(0, max(1, n_t - mask_size))
    data[:, start:start + mask_size] = 0
    return data


def time_mask_augment(filepaths, mask_size=900):
    X_aug = []

    for fp in filepaths:
        raw = load_raw(fp)
        data = raw.get_data()
        sfreq = raw.info["sfreq"]

        masked = time_masking(data.copy(), mask_size)
        feat = extract_band_power_from_array(masked, sfreq)

        X_aug.append(feat)

    return np.vstack(X_aug)


# -------- feature-level --------

def g_gaussian_noise(X, noise_level=0.01):
    return X + np.random.normal(0, noise_level * X.std(), X.shape)


def g_amplitude_scaling(X, scale_range=(0.9, 1.1)):
    return X * np.random.uniform(*scale_range)


def f_gaussian_noise(X, noise_std=0.01):
    return X + np.random.normal(0, noise_std, X.shape)


def f_amplitude_scaling(X, low=0.9, high=1.1):
    scales = np.random.uniform(low, high, (X.shape[0], 1))
    return X * scales


# ======================================================
# EVALUATION
# ======================================================

def run_svm_cv(name, X, y, groups, filepaths, augmenter=None):

    cv = GroupKFold(n_splits=N_SPLITS)

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    acc, f1, roc = [], [], []

    for train_idx, test_idx in cv.split(X, y, groups):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fp_train = filepaths[train_idx]

        if augmenter is not None:
            X_aug = augmenter(X_train, fp_train)

            X_train = np.vstack([X_train, X_aug])
            y_train = np.concatenate([y_train, y_train])

        model = clone(base_model)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        acc.append(accuracy_score(y_test, pred))
        f1.append(f1_score(y_test, pred))

        if len(np.unique(y_test)) > 1:
            roc.append(roc_auc_score(y_test, prob))
        else:
            roc.append(np.nan)

    return {
        "Condition": name,
        "Accuracy": np.mean(acc),
        "F1": np.mean(f1),
        "ROC-AUC": np.nanmean(roc)
    }


# ======================================================
# AUGMENTATION MAP
# ======================================================

def get_augmentations():
    return {
        "Baseline": lambda X, fp: X,
        "Time Mask": lambda X, fp: time_mask_augment(fp),
        "Global Gaussian": lambda X, fp: g_gaussian_noise(X),
        "Global Amplitude": lambda X, fp: g_amplitude_scaling(X),
        "Feature Gaussian": lambda X, fp: f_gaussian_noise(X),
        "Feature Amplitude": lambda X, fp: f_amplitude_scaling(X),
    }


# ======================================================
# ENTRY POINT FOR run_data_augment.py
# ======================================================

def run_standard_augmentation():

    X, y, groups, filepaths = build_dataset()

    results = []

    for name, aug_fn in get_augmentations().items():
        results.append(
            run_svm_cv(name, X, y, groups, filepaths, augmenter=aug_fn)
        )

    return pd.DataFrame(results)