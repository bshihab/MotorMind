"""
EEG Feature Extraction Module

This module contains functions for extracting features from EEG data
and converting them to text representations suitable for LLM analysis.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Any, Union, Optional


def compute_band_power(eeg_data, fs, channel_names=None, window_size=None):
    """
    Compute power in different frequency bands for each channel.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        fs: Sampling frequency (Hz)
        channel_names: List of channel names
        window_size: Window size in samples (None for entire signal)
        
    Returns:
        Dictionary containing power values for each frequency band and channel
    """
    # Default channel names if not provided
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Create result dictionary
    band_powers = {}
    for band_name in bands:
        band_powers[f"{band_name}_power"] = {}
    
    # Compute power for each channel
    for i, channel in enumerate(channel_names):
        # Get channel data
        channel_data = eeg_data[i, :]
        
        # Apply window if specified
        if window_size is not None:
            if window_size <= len(channel_data):
                channel_data = channel_data[:window_size]
        
        # Compute power spectral density
        f, psd = signal.welch(channel_data, fs, nperseg=min(256, len(channel_data)))
        
        # Calculate power in each band
        for band_name, (low, high) in bands.items():
            # Find frequencies in band
            idx_band = np.logical_and(f >= low, f <= high)
            
            # Calculate average power in band
            if np.any(idx_band):
                power = np.mean(psd[idx_band])
                band_powers[f"{band_name}_power"][channel] = power
    
    return band_powers


def compute_time_domain_features(eeg_data, channel_names=None):
    """
    Compute time domain features for each channel.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        channel_names: List of channel names
        
    Returns:
        Dictionary containing time domain features for each channel
    """
    # Default channel names if not provided
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
    
    # Create result dictionary
    features = {
        'mean': {},
        'std': {},
        'kurtosis': {},
        'skewness': {},
        'line_length': {},
        'zero_crossings': {}
    }
    
    # Compute features for each channel
    for i, channel in enumerate(channel_names):
        # Get channel data
        channel_data = eeg_data[i, :]
        
        # Calculate features
        features['mean'][channel] = np.mean(channel_data)
        features['std'][channel] = np.std(channel_data)
        features['kurtosis'][channel] = stats.kurtosis(channel_data)
        features['skewness'][channel] = stats.skew(channel_data)
        
        # Line length (sum of absolute differences)
        features['line_length'][channel] = np.sum(np.abs(np.diff(channel_data)))
        
        # Zero crossings
        zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
        features['zero_crossings'][channel] = len(zero_crossings)
    
    return features


def extract_all_features(eeg_data, fs, channel_names=None, window_size=None, overlap=0):
    """
    Extract a comprehensive set of features from EEG data.
    
    Args:
        eeg_data: EEG data of shape (channels, samples)
        fs: Sampling frequency (Hz)
        channel_names: List of channel names
        window_size: Window size in samples (None for entire signal)
        overlap: Overlap between consecutive windows in samples (0 for no overlap)
        
    Returns:
        List of feature dictionaries, one per window
    """
    # Default channel names if not provided
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
    
    # If window_size is None, use the entire signal
    if window_size is None:
        window_size = eeg_data.shape[1]
    
    # Ensure window_size and overlap are integers
    window_size = int(window_size)
    overlap = int(overlap)
    
    # Calculate step size
    step = window_size - overlap
    
    # Calculate number of windows
    n_windows = (eeg_data.shape[1] - window_size) // step + 1
    
    # Extract features for each window
    features_list = []
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        
        # Extract window
        window_data = eeg_data[:, start:end]
        
        # Calculate window time in seconds
        window_start_time = start / fs
        window_end_time = end / fs
        
        # Extract features
        window_features = {
            'window_start': window_start_time,
            'window_end': window_end_time,
            'window_size': window_size / fs
        }
        
        # Frequency domain features
        band_powers = compute_band_power(window_data, fs, channel_names)
        window_features.update(band_powers)
        
        # Time domain features
        time_features = compute_time_domain_features(window_data, channel_names)
        window_features.update(time_features)
        
        # Calculate alpha/beta ratio (important for motor imagery)
        alpha_beta_ratio = {}
        for channel in channel_names:
            if channel in band_powers['alpha_power'] and channel in band_powers['beta_power']:
                alpha = band_powers['alpha_power'][channel]
                beta = band_powers['beta_power'][channel]
                if beta > 0:
                    alpha_beta_ratio[channel] = alpha / beta
                else:
                    alpha_beta_ratio[channel] = 0
        window_features['alpha_beta_ratio'] = alpha_beta_ratio
        
        # Calculate motor imagery-specific features if C3 and C4 are present
        if 'C3' in channel_names and 'C4' in channel_names:
            c3_idx = channel_names.index('C3')
            c4_idx = channel_names.index('C4')
            
            # Calculate ERD/ERS (simplified)
            # In a real implementation, this would compare to a baseline period
            window_features['erd_ers'] = {
                'C3': -10 + 20 * np.random.random(),  # Placeholder for demo
                'C4': -10 + 20 * np.random.random()   # Placeholder for demo
            }
            
            # Calculate laterality
            if 'alpha_power' in band_powers:
                if 'C3' in band_powers['alpha_power'] and 'C4' in band_powers['alpha_power']:
                    c3_alpha = band_powers['alpha_power']['C3']
                    c4_alpha = band_powers['alpha_power']['C4']
                    sum_alpha = c3_alpha + c4_alpha
                    if sum_alpha > 0:
                        laterality = (c3_alpha - c4_alpha) / sum_alpha
                    else:
                        laterality = 0
                    window_features['alpha_laterality'] = laterality
        
        features_list.append(window_features)
    
    return features_list


def features_to_text(feature_dict, task_context="EEG recording during motor imagery task"):
    """
    Convert numeric EEG features to a text representation suitable for LLM analysis.
    
    Args:
        feature_dict: Dictionary of EEG features
        task_context: Context description of the recording
        
    Returns:
        Text representation of the features
    """
    text_parts = [f"Context: {task_context}\n"]
    
    # Add recording details
    text_parts.append(f"Window: {feature_dict.get('window_start', 0):.2f}s to {feature_dict.get('window_end', 2):.2f}s\n")
    
    # Add frequency band power features
    text_parts.append("Frequency Band Power:")
    
    # Delta band (0.5-4 Hz)
    if 'delta_power' in feature_dict:
        text_parts.append("\nDelta band (0.5-4 Hz) power by channel:")
        for channel, value in feature_dict['delta_power'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV²")
    
    # Theta band (4-8 Hz)
    if 'theta_power' in feature_dict:
        text_parts.append("\nTheta band (4-8 Hz) power by channel:")
        for channel, value in feature_dict['theta_power'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV²")
    
    # Alpha band (8-13 Hz)
    if 'alpha_power' in feature_dict:
        text_parts.append("\nAlpha band (8-13 Hz) power by channel:")
        for channel, value in feature_dict['alpha_power'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV²")
    
    # Beta band (13-30 Hz)
    if 'beta_power' in feature_dict:
        text_parts.append("\nBeta band (13-30 Hz) power by channel:")
        for channel, value in feature_dict['beta_power'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV²")
    
    # Gamma band (30-45 Hz)
    if 'gamma_power' in feature_dict:
        text_parts.append("\nGamma band (30-45 Hz) power by channel:")
        for channel, value in feature_dict['gamma_power'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV²")
    
    # Add time domain features
    text_parts.append("\nTime Domain Features:")
    
    # Line length (signal complexity measure)
    if 'line_length' in feature_dict:
        text_parts.append("\nLine Length (signal complexity):")
        for channel, value in feature_dict['line_length'].items():
            text_parts.append(f"  {channel}: {value:.2f}")
    
    # Zero crossings
    if 'zero_crossings' in feature_dict:
        text_parts.append("\nZero Crossings (frequency measure):")
        for channel, value in feature_dict['zero_crossings'].items():
            text_parts.append(f"  {channel}: {value}")
    
    # Statistical measures
    if 'mean' in feature_dict:
        text_parts.append("\nMean Signal Value:")
        for channel, value in feature_dict['mean'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV")
    
    if 'std' in feature_dict:
        text_parts.append("\nSignal Standard Deviation:")
        for channel, value in feature_dict['std'].items():
            text_parts.append(f"  {channel}: {value:.2f} µV")
    
    # Add alpha/beta ratio (important for motor imagery)
    if 'alpha_beta_ratio' in feature_dict:
        text_parts.append("\nAlpha/Beta Ratio (higher values indicate relaxed state):")
        for channel, value in feature_dict['alpha_beta_ratio'].items():
            text_parts.append(f"  {channel}: {value:.2f}")
    
    # Add event-related desynchronization/synchronization if available
    if 'erd_ers' in feature_dict:
        text_parts.append("\nEvent-Related Desynchronization/Synchronization:")
        for channel, value in feature_dict['erd_ers'].items():
            direction = "Desynchronization" if value < 0 else "Synchronization"
            text_parts.append(f"  {channel}: {abs(value):.2f}% {direction}")
    
    # Add sensorimotor rhythm description (C3/C4 comparison - critical for left/right hand imagery)
    if 'alpha_laterality' in feature_dict:
        laterality = feature_dict['alpha_laterality']
        text_parts.append("\nSensorimotor Rhythm (C3 vs C4):")
        if laterality > 0.1:
            text_parts.append("  Right hemisphere shows stronger activity (C3 > C4)")
            text_parts.append("  This pattern is often associated with left hand motor imagery")
        elif laterality < -0.1:
            text_parts.append("  Left hemisphere shows stronger activity (C4 > C3)")
            text_parts.append("  This pattern is often associated with right hand motor imagery")
        else:
            text_parts.append("  Relatively balanced activity between hemispheres")
            text_parts.append("  This pattern may indicate feet movement or rest state")
    
    # For motor imagery task, add interpretation hints
    if "motor imagery" in task_context.lower():
        text_parts.append("\nMotor Imagery Interpretation Guide:")
        text_parts.append("  - Right hand imagery typically shows ERD in left sensorimotor cortex (C3)")
        text_parts.append("  - Left hand imagery typically shows ERD in right sensorimotor cortex (C4)")
        text_parts.append("  - Feet imagery typically shows ERD in central area (Cz)")
        text_parts.append("  - Rest state typically shows higher alpha power across channels")
        text_parts.append("  - ERD = Event-Related Desynchronization (power decrease)")
        text_parts.append("  - ERS = Event-Related Synchronization (power increase)")
    
    return "\n".join(text_parts) 