# scr/features.py

import numpy as np
import mne 

from .config import BANDS, COMMON_CHANNELS

def extract_band_power(raw,common_channels=COMMON_CHANNELS, bands=BANDS) -> np.ndarray:
    """
    Extract relative band-power features for each EEG channel in the raw MNE object.

    Returns
    --------
    np.ndarray
        Flattened feature vector with shape:
        n_channels * n_bands
    """

    raw_copy = raw.copy()
    raw_copy.pick(common_channels)
    
    # get data 
    data = raw_copy.get_data()      # shape: (n_channels, n_time points)
    sfreq = raw_copy.info['sfreq']  # sampling frequency (e.g. 256 Hz)

    return extract_band_power_from_array(data, sfreq, bands=bands)
  

def extract_band_power_from_array(epoch_data, sfreq, bands=BANDS) -> np.ndarray:
    """ 
    Extract relative band-power features from one EEG segment. 

    Parameters
    ----------
    epoch_data: np.ndarray
        Shape: (n_channels, n_timepoints)

    sfreq: float
        Sampling frequency.

    bands: dict
        Frequency bands to extract.

    Returns
    ---------
      Flattened feature vector with shape:
      n_channels * n_bands
    """

    all_features = []

    for ch in epoch_data:
        n_fft = min(len(ch), 256)
        n_per_seg = min(len(ch), 256)
        
        psd, freqs = mne.time_frequency.psd_array_welch(
            ch,
            sfreq=sfreq,
            fmin=1,
            fmax=30,
            n_fft=n_fft,   
            n_per_seg=n_per_seg,
            verbose=False
        )

        total_power = psd.sum()

        band_features = [
            (
                psd[(freqs >= fmin) & (freqs <= fmax)].mean() / total_power
                if total_power > 0 else 0 
            )
            for (fmin, fmax) in bands.values()
        ]

        all_features.append(band_features)

    return np.nan_to_num(np.array(all_features).flatten())

def zscore_segments(X):
    """
    Z-score each channel within each sample. 

    Input shape
    ----------
    X: np.ndarray
        Shape: (n_samples, n_channels, n_timepoints)

    Returns
    -------
    np.ndarray
        Z-scored array with same shape.
    """
    X = X.astype(np.float32, copy=False)

    mean = X.mean(axis=2, keepdims=True).astype(np.float32)
    std = X.std(axis=2, keepdims=True).astype(np.float32)

    std[std == 0] = 1.0

    X -= mean
    X /= std

    return X 

def prepare_cnn_input(X):
    """
    Convert raw EEG windows into Keras Conv1D input shape.
    
    Input:
        (samples, channels, timepoints)
        
    Output:
        (samples, timeponts, channels)
    """

    X_z = zscore_segments(X.copy())
    return np.transpose(X_z, (0,2,1))