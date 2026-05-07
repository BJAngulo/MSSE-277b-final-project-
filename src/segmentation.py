# src/segmentation.py

import numpy as np 
import pandas as pd
import mne 

from .config import SEGMENTED_DATA_DIR
from .features import extract_band_power_from_array

# =========================
# Welch
# =========================

def build_segmented_dataset(metadata_df, common_channels, window_sec=5.0):
    """Build segmented Welch band-power features from EDF recordings."""
    X_segments = []
    y_segments = []
    groups_segments = []
    condition_segments = []
    recording_segments = []
    segment_indices = []

    for _, row in metadata_df.iterrows():
        raw = mne.io.read_raw_edf(row["filepath"], preload=True, verbose=False)
        raw.pick(common_channels)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=window_sec,
            preload=True,
            overlap=0.0,
            verbose=False
        )

        epoch_data = epochs.get_data()
        sfreq = epochs.info["sfreq"]

        for i, segment in enumerate(epoch_data):
            features = extract_band_power_from_array(segment, sfreq)

            X_segments.append(features)
            y_segments.append(row["label"])
            groups_segments.append(row["patient_id"])
            condition_segments.append(row["condition"])
            recording_segments.append(f"{row['recording_id']}_seg{i}")
            segment_indices.append(i)

    X_segments = np.vstack(X_segments)
    y_segments = np.array(y_segments)
    groups_segments = np.array(groups_segments)

    segmented_df = pd.DataFrame({
        "recording_segment": recording_segments,
        "patient_id": groups_segments,
        "label": y_segments,
        "condition": condition_segments,
        "segment_idx": segment_indices
    })

    return X_segments, y_segments, groups_segments, segmented_df

def save_segmented_data(X, y, groups, seg_df, window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Save Welch segmented features and metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(window_sec)}sec"

    np.save(save_dir / f"X_seg_{suffix}.npy", X)
    np.save(save_dir / f"y_seg_{suffix}.npy", y)
    np.save(save_dir / f"groups_seg_{suffix}.npy", groups)
    seg_df.to_csv(save_dir / f"seg_df_{suffix}.csv", index=False)

def load_segmented_data(window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Load cached Welch segmented features."""
    suffix = f"{int(window_sec)}sec"

    X = np.load(save_dir /  f"X_seg_{suffix}.npy")
    y = np.load(save_dir / f"y_seg_{suffix}.npy")
    groups = np.load(save_dir / f"groups_seg_{suffix}.npy")
    seg_df = pd.read_csv(save_dir / f"seg_df_{suffix}.csv")

    return X, y, groups, seg_df

def get_or_build_segmented_data(metadata_df, common_channels, window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Load cached Welch features if available; otherwise build and save."""
    suffix = f"{int(window_sec)}sec"
    X_path = save_dir / f"X_seg_{suffix}.npy"

    if X_path.exists():
        print(f"loading cached data for {window_sec}s...")
        return load_segmented_data(window_sec, save_dir=save_dir)
    
    print(f"Building Welch segmented data for {window_sec}s...")

    X, y, groups, seg_df = build_segmented_dataset(
        metadata_df, 
        common_channels,
        window_sec=window_sec
    )

    save_segmented_data(X, y, groups, seg_df, window_sec, save_dir=save_dir)

    return X, y, groups, seg_df 

# ==============================
# RAW
# ==============================

def build_raw_segmented_dataset(metadata_df, common_channels, window_sec=10.0):
    """ Build raw EEG windows without Welch feature extraction."""
    X_segments = []
    y_segments = []
    groups_segments = []
    condition_segments = []
    recording_segments = []
    segment_indices = []
        
    for _, row in metadata_df.iterrows():
        raw = mne.io.read_raw_edf(row["filepath"], preload=True, verbose=False)
        raw.pick(common_channels)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=window_sec,
            preload=True,
            overlap=0.0,
            verbose=False
        )

        epoch_data = epochs.get_data() 

        for i, segment in enumerate(epoch_data):
            X_segments.append(segment)
            y_segments.append(row["label"])
            groups_segments.append(row["patient_id"])
            condition_segments.append(row["condition"])
            recording_segments.append(f"{row['recording_id']}_seg{i}")
            segment_indices.append(i)

    X_segments = np.stack(X_segments)
    y_segments = np.array(y_segments)
    groups_segments = np.array(groups_segments)

    raw_seg_df = pd.DataFrame({
        "recording_segment": recording_segments,
        "patient_id": groups_segments,
        "label": y_segments,
        "condition": condition_segments,
        "segment_idx": segment_indices
    })

    return X_segments, y_segments, groups_segments, raw_seg_df 

def save_raw_segmented_data(X, y, groups, seg_df, window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Save raw segmented EEG data to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)

    suffix=f"{int(window_sec)}sec"

    np.save(save_dir / f"X_raw_{suffix}.npy", X)
    np.save(save_dir / f"y_raw_{suffix}.npy", y)
    np.save(save_dir / f"groups_raw_{suffix}.npy", groups)
    seg_df.to_csv(save_dir / f"raw_seg_df_{suffix}.csv", index=False)

def load_raw_segmented_data(window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Load cached raw segmented EEG data."""
    suffix = f"{int(window_sec)}sec"

    X = np.load(save_dir / f"X_raw_{suffix}.npy")
    y = np.load(save_dir / f"y_raw_{suffix}.npy")
    groups = np.load(save_dir / f"groups_raw_{suffix}.npy")
    seg_df = pd.read_csv(save_dir / f"raw_seg_df_{suffix}.csv")

    return X, y, groups, seg_df

def get_or_build_raw_segmented_data(metadata_df, common_channels, window_sec, save_dir=SEGMENTED_DATA_DIR):
    """ Load cached raw EEG segments if available; otherwise build and save."""
    suffix = f"{int(window_sec)}sec"
    X_path = save_dir / f"X_raw_{suffix}.npy"

    if X_path.exists():
        print(f"Loading cached raw data for {window_sec}s...")
        return load_raw_segmented_data(window_sec, save_dir=save_dir)
    
    print(f"Building Raw segmented data for {window_sec}s...")
    X, y, groups, seg_df = build_raw_segmented_dataset(
        metadata_df,
        common_channels,
        window_sec=window_sec
    )

    save_raw_segmented_data(X, y, groups, seg_df, window_sec, save_dir=save_dir)

    return X, y, groups, seg_df