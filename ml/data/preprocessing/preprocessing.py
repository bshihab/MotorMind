"""
EEG Preprocessing Module

This module contains functions for preprocessing EEG data, including:
- Filtering (bandpass, notch)
- Normalization
- Segmentation
- Artifact removal
"""

import numpy as np
from scipy import signal


def bandpass_filter(eeg_data, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to EEG data.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered EEG data of same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Create Butterworth filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter along the samples axis (axis 1)
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        filtered_data[i, :] = signal.filtfilt(b, a, eeg_data[i, :])
    
    return filtered_data


def notch_filter(eeg_data, notch_freq, fs, quality_factor=30):
    """
    Apply a notch filter to remove power line interference.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        notch_freq: Frequency to remove (typically 50 or 60 Hz)
        fs: Sampling frequency (Hz)
        quality_factor: Quality factor of the notch filter
        
    Returns:
        Filtered EEG data of same shape as input
    """
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    
    # Apply filter along the samples axis (axis 1)
    filtered_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        filtered_data[i, :] = signal.filtfilt(b, a, eeg_data[i, :])
    
    return filtered_data


def normalize_eeg(eeg_data, method='zscore'):
    """
    Normalize EEG data using specified method.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        method: Normalization method ('zscore', 'minmax', or 'robust')
        
    Returns:
        Normalized EEG data of same shape as input
    """
    normalized_data = np.zeros_like(eeg_data)
    
    if method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        for i in range(eeg_data.shape[0]):
            channel_data = eeg_data[i, :]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 0:  # Avoid division by zero
                normalized_data[i, :] = (channel_data - mean) / std
            else:
                normalized_data[i, :] = channel_data - mean
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        for i in range(eeg_data.shape[0]):
            channel_data = eeg_data[i, :]
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            if max_val > min_val:  # Avoid division by zero
                normalized_data[i, :] = (channel_data - min_val) / (max_val - min_val)
            else:
                normalized_data[i, :] = np.zeros_like(channel_data)
    
    elif method == 'robust':
        # Robust normalization using median and IQR
        for i in range(eeg_data.shape[0]):
            channel_data = eeg_data[i, :]
            median = np.median(channel_data)
            q75, q25 = np.percentile(channel_data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:  # Avoid division by zero
                normalized_data[i, :] = (channel_data - median) / iqr
            else:
                normalized_data[i, :] = channel_data - median
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data


def segment_eeg(eeg_data, window_size, overlap=0, fs=None):
    """
    Segment EEG data into windows.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        window_size: Window size in samples, or in seconds if fs is provided
        overlap: Overlap between consecutive windows in samples, or in seconds if fs is provided
        fs: Sampling frequency (Hz), if provided, window_size and overlap are in seconds
        
    Returns:
        Segmented EEG data of shape (windows, channels, samples_per_window)
    """
    n_channels, n_samples = eeg_data.shape
    
    # Convert time to samples if fs is provided
    if fs is not None:
        window_size = int(window_size * fs)
        overlap = int(overlap * fs)
    
    # Calculate step size
    step = window_size - overlap
    
    # Calculate number of windows
    n_windows = (n_samples - window_size) // step + 1
    
    # Initialize segmented data
    segmented_data = np.zeros((n_windows, n_channels, window_size))
    
    # Extract windows
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        segmented_data[i, :, :] = eeg_data[:, start:end]
    
    return segmented_data


def remove_artifacts(eeg_data, threshold=100, method='clip'):
    """
    Remove artifacts from EEG data.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        threshold: Threshold for artifact detection, in µV or z-score
        method: Method for artifact removal ('clip', 'zero', or 'interp')
        
    Returns:
        Cleaned EEG data of same shape as input
    """
    cleaned_data = eeg_data.copy()
    
    for i in range(eeg_data.shape[0]):
        channel_data = eeg_data[i, :]
        
        # Detect artifacts
        if method == 'clip':
            # Clip values beyond threshold
            cleaned_data[i, :] = np.clip(channel_data, -threshold, threshold)
        
        elif method == 'zero':
            # Zero out values beyond threshold
            mask = np.abs(channel_data) > threshold
            cleaned_data[i, mask] = 0
        
        elif method == 'interp':
            # Replace values beyond threshold with interpolated values
            mask = np.abs(channel_data) > threshold
            indices = np.arange(len(channel_data))
            valid_indices = indices[~mask]
            valid_values = channel_data[~mask]
            
            if len(valid_values) > 0:  # Only interpolate if there are valid values
                interpolated = np.interp(indices[mask], valid_indices, valid_values)
                cleaned_data[i, mask] = interpolated
        
        else:
            raise ValueError(f"Unknown artifact removal method: {method}")
    
    return cleaned_data


def preprocess_eeg(eeg_data, fs, lowcut=5, highcut=40, notch=50, normalize=True, norm_method='zscore'):
    """
    Apply a standard preprocessing pipeline to EEG data.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        fs: Sampling frequency (Hz)
        lowcut: Lower cutoff frequency (Hz)
        highcut: Upper cutoff frequency (Hz)
        notch: Notch filter frequency (Hz), set to None to disable
        normalize: Whether to normalize the data
        norm_method: Normalization method if normalize is True
        
    Returns:
        Preprocessed EEG data of same shape as input
    """
    # Apply bandpass filter
    filtered_data = bandpass_filter(eeg_data, lowcut, highcut, fs)
    
    # Apply notch filter if specified
    if notch is not None:
        filtered_data = notch_filter(filtered_data, notch, fs)
    
    # Normalize if specified
    if normalize:
        preprocessed_data = normalize_eeg(filtered_data, method=norm_method)
    else:
        preprocessed_data = filtered_data
    
    return preprocessed_data 