# scr.run_analysis.py 

import pandas as pd

from .config import(
    COMMON_CHANNELS,
    WELCH_WINDOW_SIZES,
    RAW_WINDOW_SIZE,
    N_SPLITS,
    TABLES_DIR,
)

from .metadata import build_metadata
from .segmentation import (
    get_or_build_segmented_data,
    get_or_build_raw_segmented_data,
)
from .data_loading import load_window_results
from .deep_learning import get_cached_deep_comparison

def build_classical_results_df(window_results_df, raw_pca_df):
    """ Build classical model comparison rows from cached result tables."""

    window_results_df = window_results_df.rename(columns={"roc-auc": "roc_auc"})
    raw_pca_df = raw_pca_df.rename(columns={"roc-auc": "roc_auc"})

    welch_svm_10 = window_results_df[
        (window_results_df["window_sec"] == 10.0)
        & (window_results_df["model_name"] == "SVM RBF")
    ].iloc[0]

    raw_pca_10 = raw_pca_df.iloc[0]

    classical_results_df = pd.DataFrame([
        {
            "method": "Welch + SVM",
            "model_name": "SVM (RBF)",
            "window_sec": 10,
            "accuracy": welch_svm_10["accuracy"],
            "f1": welch_svm_10["f1"],
            "roc_auc": welch_svm_10["roc_auc"],
            "n_folds": welch_svm_10.get("n_folds", N_SPLITS),
            "notes": "Interpretable Welch relative band-power baseline",
        },
        {
           "method": "PCA + SVM",
            "model_name": "PCA + SVM",
            "window_sec": 10,
            "accuracy": raw_pca_10["accuracy"],
            "f1": raw_pca_10["f1"],
            "roc_auc": raw_pca_10["roc_auc"],
            "n_folds": welch_svm_10.get("n_folds", N_SPLITS),
            "notes": "Flattened raw EEG with PCA before SVM", 
        },
    ])

    return classical_results_df

def build_unified_model_comparison(classical_results_df, cached_deep_comparison_df):
    """Combine classical and cached deep learning results."""

    unified_df = pd.concat(
        [classical_results_df, cached_deep_comparison_df],
        ignore_index=True,
    )

    unified_df = unified_df.rename(columns={"roc-auc": "roc_auc"})

    if "n-folds" in unified_df.columns:
        unified_df["n_folds"] = unified_df["n_folds"].fillna(N_SPLITS)
    else:
        unified_df["n-folds"] = N_SPLITS

    unified_df = unified_df.sort_values(
        by="roc_auc",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return unified_df

def main():
    print("Starting cached EEG-MDD analysis pipeline...")

    # 1. Build metadata
    metadata_df = build_metadata()
    print(f"Metadata shape: {metadata_df.shape}")

    # 2. Ensure Welch segmented datasets exist
    for window_sec in WELCH_WINDOW_SIZES:
        X, y, groups, seg_df = get_or_build_segmented_data(
            metadata_df,
            COMMON_CHANNELS, 
            window_sec=window_sec,
        )
        print(f"Welch {window_sec}s shape: {X.shape}")

    # 3. Ensure raw segmented dataset exists
    X_raw, y_raw, groups_raw, raw_seg_df = get_or_build_raw_segmented_data(
        metadata_df,
        COMMON_CHANNELS,
        window_sec=RAW_WINDOW_SIZE,
    )
    print(f"Raw {RAW_WINDOW_SIZE}s shape: {X_raw.shape}")

    # 4. Load cacged result tables
    window_results_df = load_window_results()
    raw_pca_df = pd.read_csv(TABLES_DIR / "raw_pca_svm_10sec.csv")
    cached_deep_comparison_df = get_cached_deep_comparison()

    # 5. Build unified comparison 
    classical_results_df = build_classical_results_df(
        window_results_df,
        raw_pca_df,
    )

    unified_df = build_unified_model_comparison(
        classical_results_df,
        cached_deep_comparison_df,
    )

    # 6. Save final comparison 
    out_path = TABLES_DIR / "unified_model_comparison.csv"
    unified_df.to_csv(out_path, index=False)

    print(f"Saved unified comparison to: {out_path}")
    print(unified_df)


if __name__ == "__main__":
    main()
    



