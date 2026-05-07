# scr/evaluation.py

import numpy as np 
from sklearn.model_selection import GroupKFold, cross_validate

from .config import N_SPLITS

def evaluate_sklearn_model(model, X, y, groups, n_splits=N_SPLITS):
    """
    Evaluate a sklearn-compatible model using GroupKFold.
    
    Prevents patient-level leakage by keeping all samples from the same patient in the same fold.
    """

    cv = GroupKFold(n_splits=n_splits)

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        model,
        X,
        y, 
        groups=groups,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    return {
        "accuracy": np.mean(scores["test_accuracy"]),
        "f1": np.mean(scores["test_f1"]),
        "roc-auc": np.mean(scores["test_roc_auc"]),
        "accuracy_std": np.std(scores["test_accuracy"]),
        "f1_std": np.std(scores["test_f1"]),
        "roc_auc_std": np.std(scores["test_roc_auc"]),
        "n-folds": n_splits,

    }
