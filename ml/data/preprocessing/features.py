"""
EEG Feature Extraction Module

This module provides functions for extracting features from EEG data that can
be used as input to Large Language Models for analysis and interpretation.

Key features extracted:
- Line length features
- Power spectral features
- Statistical features
- Connectivity measures
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union


def extract_line_length(eeg_data: np.ndarray, window_size: int = 100) -> np.ndarray:
    """
    Calculate the line length feature for each channel of EEG data.
    Line length is the sum of absolute differences between consecutive samples.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        window_size: Size of the window for line length calculation
        
    Returns:
        Line length features with shape (channels, n_windows)
    """
    n_channels, n_samples = eeg_data.shape
    n_windows = n_samples // window_size
    
    line_length = np.zeros((n_channels, n_windows))
    
    for ch in range(n_channels):
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window_data = eeg_data[ch, start:end]
            
            # Calculate line length as sum of absolute differences
            ll = np.sum(np.abs(np.diff(window_data)))
            line_length[ch, w] = ll
            
    return line_length


def extract_power_spectral_features(
    eeg_data: np.ndarray, 
    fs: int, 
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract power spectral features for specified frequency bands.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        bands: Dictionary of frequency bands {name: (low_freq, high_freq)}
            Default bands: delta, theta, alpha, beta, gamma
            
    Returns:
        Dictionary with band powers for each channel
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    n_channels = eeg_data.shape[0]
    result = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Initialize band power array
        band_power = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Calculate power spectral density
            f, psd = signal.welch(eeg_data[ch], fs=fs, nperseg=min(256, len(eeg_data[ch])))
            
            # Find frequency band indices
            idx_band = np.logical_and(f >= low_freq, f <= high_freq)
            
            # Calculate average power in the band
            band_power[ch] = np.mean(psd[idx_band])
            
        result[band_name] = band_power
        
    return result


def extract_statistical_features(eeg_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract statistical features from EEG data.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        
    Returns:
        Dictionary of statistical features for each channel
    """
    n_channels = eeg_data.shape[0]
    
    # Initialize feature arrays
    variance = np.zeros(n_channels)
    skewness = np.zeros(n_channels)
    kurt = np.zeros(n_channels)
    
    for ch in range(n_channels):
        variance[ch] = np.var(eeg_data[ch])
        skewness[ch] = skew(eeg_data[ch])
        kurt[ch] = kurtosis(eeg_data[ch])
    
    return {
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurt
    }


def extract_connectivity(
    eeg_data: np.ndarray, 
    fs: int, 
    method: str = 'phase_sync',
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate connectivity measures between EEG channels.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        method: Connectivity method ('phase_sync', 'correlation')
        bands: Dictionary of frequency bands for filtering
        
    Returns:
        Connectivity matrices for each band with shape (channels, channels)
    """
    if bands is None:
        bands = {
            'alpha': (8, 13),
            'beta': (13, 30)
        }
    
    n_channels = eeg_data.shape[0]
    result = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        # Initialize connectivity matrix
        conn_matrix = np.zeros((n_channels, n_channels))
        
        # Apply bandpass filter
        b, a = signal.butter(3, [low_freq/(fs/2), high_freq/(fs/2)], btype='bandpass')
        filtered_data = np.array([signal.filtfilt(b, a, eeg_data[ch]) for ch in range(n_channels)])
        
        if method == 'phase_sync':
            # Calculate phase using Hilbert transform
            analytic_signal = signal.hilbert(filtered_data)
            phase = np.angle(analytic_signal)
            
            # Calculate phase synchronization
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    phase_diff = phase[i] - phase[j]
                    # Phase Locking Value
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    conn_matrix[i, j] = plv
                    conn_matrix[j, i] = plv
        
        elif method == 'correlation':
            # Calculate correlation matrix
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    corr = np.corrcoef(filtered_data[i], filtered_data[j])[0, 1]
                    conn_matrix[i, j] = corr
                    conn_matrix[j, i] = corr
        
        result[band_name] = conn_matrix
        
    return result


def extract_all_features(
    eeg_data: np.ndarray,
    fs: int,
    channel_names: List[str],
    window_size: int = 100,
    overlap: float = 0.5
) -> List[Dict]:
    """
    Extract all features from EEG data using sliding windows.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        channel_names: List of channel names
        window_size: Window size in samples
        overlap: Window overlap ratio (0-1)
        
    Returns:
        List of feature dictionaries for each window
    """
    n_channels, n_samples = eeg_data.shape
    step_size = int(window_size * (1 - overlap))
    n_windows = (n_samples - window_size) // step_size + 1
    
    all_features = []
    
    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        
        window_data = eeg_data[:, start:end]
        
        # Extract features
        line_length = extract_line_length(window_data, window_size=window_size)[:, 0]
        
        power_features = extract_power_spectral_features(window_data, fs)
        
        stat_features = extract_statistical_features(window_data)
        
        connectivity = extract_connectivity(
            window_data, 
            fs, 
            method='phase_sync',
            bands={'alpha': (8, 13), 'beta': (13, 30)}
        )
        
        # Collect all features
        window_features = {
            'window_start': start,
            'window_end': end,
            'line_length': {channel_names[ch]: float(line_length[ch]) for ch in range(n_channels)},
            'power_spectral': {
                band_name: {channel_names[ch]: float(powers[ch]) for ch in range(n_channels)}
                for band_name, powers in power_features.items()
            },
            'statistical': {
                feat_name: {channel_names[ch]: float(values[ch]) for ch in range(n_channels)}
                for feat_name, values in stat_features.items()
            },
            'connectivity': {
                band_name: {
                    f"{channel_names[i]}-{channel_names[j]}": float(matrix[i, j])
                    for i in range(n_channels) for j in range(i+1, n_channels)
                }
                for band_name, matrix in connectivity.items()
            }
        }
        
        all_features.append(window_features)
    
    return all_features


def features_to_text(features: Dict, task_context: Optional[str] = None) -> str:
    """
    Convert extracted EEG features to a text format suitable for LLM processing.
    
    Args:
        features: Dictionary of extracted features
        task_context: Optional context about the task being performed
        
    Returns:
        Formatted text representation of the features
    """
    text_parts = []
    
    if task_context:
        text_parts.append(f"Task Context: {task_context}\n")
    
    # Add window information
    text_parts.append(f"Window: {features['window_start']} to {features['window_end']} samples\n")
    
    # Add line length features
    text_parts.append("Line Length Features:")
    for channel, value in features['line_length'].items():
        text_parts.append(f"  Channel {channel}: {value:.4f}")
    text_parts.append("")
    
    # Add power spectral features
    text_parts.append("Spectral Power Features:")
    for band, channel_values in features['power_spectral'].items():
        text_parts.append(f"  {band.capitalize()} Band:")
        for channel, value in channel_values.items():
            text_parts.append(f"    Channel {channel}: {value:.4f} μV²")
    text_parts.append("")
    
    # Add statistical features
    text_parts.append("Statistical Features:")
    for stat, channel_values in features['statistical'].items():
        text_parts.append(f"  {stat.capitalize()}:")
        for channel, value in channel_values.items():
            text_parts.append(f"    Channel {channel}: {value:.4f}")
    text_parts.append("")
    
    # Add connectivity features
    text_parts.append("Connectivity Features:")
    for band, conn_values in features['connectivity'].items():
        text_parts.append(f"  {band.capitalize()} Band:")
        for channel_pair, value in conn_values.items():
            text_parts.append(f"    {channel_pair}: {value:.4f}")
    
    return "\n".join(text_parts) 