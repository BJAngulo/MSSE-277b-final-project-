import numpy as np
import pandas as pd 

from src.config import SEGMENTED_DATA_DIR, RESULTS_DIR

def load_raw_segments(window_sec=10):
    """
    Load cached raw EEG segments
    
    Returns
    -------
    X_raw : np.ndarray
    y_raw : np.ndarray
    groups_raw : np.ndarray
    segmented_raw_df : pd.DataFrame
    """

    suffix = f"{int(window_sec)}sec"

    X_raw = np.load(SEGMENTED_DATA_DIR / f"X_raw_{suffix}.npy")
    y_raw = np.load(SEGMENTED_DATA_DIR / f"y_raw_{suffix}.npy")
    groups_raw = np.load(SEGMENTED_DATA_DIR / f"groups_raw{suffix}.npy")
    segmented_raw_df = pd.read_csv(SEGMENTED_DATA_DIR / f"raw_seg_df_{suffix}.csv")

    return X_raw, y_raw, groups_raw, segmented_raw_df

def load_welch_segments(window_sec=10):
    """
    Load cached Welch-based segmented features.
    """
    suffix = f"{int(window_sec)}sec"

    X_seg = np.load(SEGMENTED_DATA_DIR / f"X_seg_{suffix}.npy")
    y_seg = np.load(SEGMENTED_DATA_DIR / f"y_seg_{suffix}.npy")
    groups_seg = np.load(SEGMENTED_DATA_DIR / f"groups_seg_{suffix}.npy")
    seg_df = pd.read_csv(SEGMENTED_DATA_DIR / f"seg_df{suffix}.csv")

    return X_seg, y_seg, groups_seg, seg_df

def load_window_results():
    """
    Load saved window size comparison results.
    """
    return pd.read_csv(RESULTS_DIR / "window_size_results.csv")

def load_temporal_ablation_results():
    """
    Load saved temporal ablation results.
    """
    ablation_results_df = pd.read_csv(RESULTS_DIR / "temporal_ablation_10sec.csv")
    baseline_ablation_df = pd.read_csv(RESULTS_DIR / "baseline_ablation_1-sec.csv")

    return baseline_ablation_df, ablation_results_df

def load_lstm_results():
    """
    Load saved LSTM fold-level and summray results.
    """
    lstm_results_df = pd.read_csv(RESULTS_DIR / "lstm_results.csv")
    lstm_summary_df = pd.read_csv(RESULTS_DIR / "lstm_summary.csv")

    return lstm_results_df, lstm_summary_df 
