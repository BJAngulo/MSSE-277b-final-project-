import numpy as np
import mne 

def extract_band_power(raw, COMMON_CHANNELS) -> np.ndarray:
    """
    Extract relative band-power features for each EEG channel in the raw MNE object.
    Returns a feature vector.
    """

    data_raw = raw.copy()
    data_raw.pick(COMMON_CHANNELS)
    
    # get data 
    data = data_raw.get_data()      # shape: (n_channels, n_time points)
    sfreq = data_raw.info['sfreq']  # sampling frequency (e.g. 256 Hz)

    # Define frequency bands 
    bands = {
        "delta": (1, 4),     # deep sleep
        "theta": (4, 8),     # drowsiness
        "alpha": (8, 13),    # relaxed 
        "beta":  (13, 30)    # active thinking 
    }

    all_features = []

    # Loop through each channel (ch) and convert signal to frequency domain using Welch's method to estimate power spectral density (psd)
    for ch in data:        
        psd, freqs = mne.time_frequency.psd_array_welch(
            ch,
            sfreq=sfreq,
            fmin=1,
            fmax=30,
            verbose=False
        )

        total_power = psd.sum()
        
        band_features = [
            (
                psd[(freqs >= fmin) & (freqs <= fmax)].mean()
            / total_power                                      # normalize by total power to get relative power in each band
            if total_power > 0 else 0                          # handle case where total power is zero to avoid division by zero
            )                           
            for (fmin, fmax) in bands.values()
            ]
    
        all_features.append(band_features)

    return np.nan_to_num(np.array(all_features).flatten())

def extract_band_power_from_array(epoch_data, sfreq):
    """ 
    epoch_data: shape (n_channels, n_timepoints)
    sfreq: sampling frequency 
    returns: flatten feature vector
    """

    bands = {
        "delta": (1, 4),     # deep sleep
        "theta": (4, 8),     # drowsiness
        "alpha": (8, 13),    # relaxed 
        "beta":  (13, 30)    # active thinking
    }

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
