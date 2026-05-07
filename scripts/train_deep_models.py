# scripts/train_deep_models.py

import sys
from pathlib import Path

# Allow imports from project root when running script directly 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import TABLES_DIR, SEGMENTED_DATA_DIR, N_SPLITS
from src.data_loading import load_raw_segments
from src.features import prepare_cnn_input 

def summarize_fold_results(results_df, model_name, method, window_sec, notes):
    """ Convert fold-level deep learning results into one summary row."""
    return {
        "method": method, 
        "model_name": model_name,
        "window_sec": window_sec,
        "accuracy": results_df["accuracy"].mean(),
        "f1": results_df["f1"].mean(),
        "roc_auc": results_df["roc_auc"].mean(),
        "n_fold": N_SPLITS,
        "notes": notes,
    }

def load_existing_deep_results():
    """Load already cached deep model fold-lvel results."""
    result_files = {
        "baseline_cnn": "baseline_cnn_results.csv",
        "deep cnn": "deep_cnn_results.csv",
        "multiscale_cnn": "multiscale_cnn_results.csv",
        "lstm_10sec": "lstm_10sec_results.csv",
        "lstm_15sec": "lstm_15sec_results.csv",
        "cnn_lstm": "cnn_lstm_results.csv"
    }

    dfs = {}

    for key, filename in result_files.items():
        path = TABLES_DIR / filename

        if path.exists():
            dfs[key] = pd.read_csv(path)
            print(f"Loaded {filename}")
        else: 
            print(f"Missing: {filename}")

    return dfs

def build_cached_deep_comparison():
    """ Build cached deep comparison table from saved fold-level result files."""
    dfs = load_existing_deep_results()

    rows= []

    if "multiscale_cnn" in dfs:
        rows.append(summarize_fold_results(
            dfs["multiscale_cnn"],
            model_name="Raw + Multiscale CNN",
            method="Raw + Multi-Scale CNN (10 sec)",
            window_sec=10,
            notes="Multi-branch Conv1D kernels on z-scored raw EEG 10-sec segments",
        ))

    if "baseline_cnn" in dfs:
         rows.append(summarize_fold_results(
            dfs["baseline_cnn"],
            model_name="Raw + Baseline CNN",
            method="Raw + Baseline CNN (10 sec)",
            window_sec=10,
            notes="Baseline Conv1D model on z-scored raw EEG 10-sec segments",
         ))

    if "deep_cnn" in dfs:
         rows.append(summarize_fold_results(
            dfs["deep_cnn"],
            model_name="Raw + Deep CNN",
            method="Raw + Deep CNN (10 sec)",
            window_sec=10,
            notes="Deeper Conv1D model on z-scored raw EEG 10-sec segments",
        ))

    if "cnn_lstm" in dfs:
         rows.append(summarize_fold_results(
            dfs["cnn_lstm"],
            model_name="Raw + CNN-LSTM",
            method="Raw + CNN-LSTM (10 sec)",
            window_sec=10,
            notes="CNN feature extractor followed by LSTM on raw EEG 10-sec segments",
        ))
         
    if "lstm_10sec" in dfs:
         rows.append(summarize_fold_results(
            dfs["lstm_10sec"],
            model_name="Raw + LSTM",
            method="Raw + LSTM (10 sec)",
            window_sec=10,
            notes="LSTM model on z-scored raw EEG 10-sec segments",
        ))
         
    if "lstm_15sec" in dfs:
         rows.append(summarize_fold_results(
            dfs["lstm_15sec"],
            model_name="Raw + LSTM",
            method="Raw + LSTM (15 sec)",
            window_sec=10,
            notes="LSTM model on z-scored raw EEG 15-sec segments",
        ))
         
    deep_comparison_df = pd.DataFrame(rows)

    out_path = TABLES_DIR / "cached_deep_comparison.csv"
    deep_comparison_df.to_csv(out_path, index=False)

    print(f"\nSaved deep comparison table to: {out_path}")
    print(deep_comparison_df)

    return deep_comparison_df 

def main():
    print("Building cached deep learning comparison table...")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    build_cached_deep_comparison()

if __name__ == "__main__":
    main()