# src/models.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import RANDOM_STATE

# =====================
# Welch Feature Models
# =====================

def get_welch_models():
    """
    Returns a dictionary of classical models for Welch features.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
    ]),

        "SVM RBF":  Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True, 
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# ===================
# Raw + PCA + SVM
# ===================

def get_raw_pca_svm(n_components=50):
    """
    Raw EGG baseline using PCA + SVM.
    Input must flattened before passing.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            random_state=RANDOM_STATE
        )),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    return model

# ===================
# Utility
# ==================

def flatten_raw(X):
    """
    Flatten raw EEG:
    (samples, channels, timepoints) -> (samples, features)
    """
    return X.reshape(X.shape[0], -1)