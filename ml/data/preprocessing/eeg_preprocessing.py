"""
EEG Preprocessing Module

This module provides functions for preprocessing EEG data specifically for 
motor imagery classification. It includes functions for:
1. Bandpass filtering
2. Signal segmentation (windowing)
3. Normalization
4. Dataset partitioning
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd


def load_eeg_numpy(file_path: str) -> np.ndarray:
    """
    Load EEG data from numpy file.
    
    Args:
        file_path: Path to the numpy file containing EEG data
        
    Returns:
        EEG data as numpy array
    """
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        return None


def bandpass_filter(
    eeg_data: np.ndarray, 
    lowcut: float = 5, 
    highcut: float = 40, 
    fs: float = 250, 
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Args:
        eeg_data: EEG data with shape (channels, samples) or (samples, channels)
        lowcut: Lower frequency bound in Hz
        highcut: Upper frequency bound in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered EEG data with same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Determine the shape and apply filter accordingly
    if eeg_data.shape[0] > eeg_data.shape[1]:  # More rows than columns (samples, channels)
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered_data[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
    else:  # More columns than rows (channels, samples)
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, eeg_data[ch, :])
            
    return filtered_data


def segment_eeg(
    eeg_data: np.ndarray, 
    fs: float = 250, 
    window_size: float = 1.0, 
    window_shift: float = 0.1
) -> List[np.ndarray]:
    """
    Segment EEG data using sliding window.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        window_size: Window size in seconds
        window_shift: Window shift in seconds
        
    Returns:
        List of EEG segments with shape (channels, samples_per_window)
    """
    w_size = int(fs * window_size)
    w_shift = int(fs * window_shift)
    segments = []
    
    i = 0
    while i + w_size <= eeg_data.shape[1]:  # Assuming (channels, samples) shape
        segments.append(eeg_data[:, i:i + w_size])
        i += w_shift
        
    return segments


def normalize_eeg(eeg_data: np.ndarray) -> np.ndarray:
    """
    Apply channel-wise z-score normalization.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        
    Returns:
        Normalized EEG data with same shape
    """
    if eeg_data.ndim == 2:  # Single trial
        return (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / (eeg_data.std(axis=1, keepdims=True) + 1e-8)
    elif eeg_data.ndim == 3:  # Multiple trials with shape (trials, channels, samples)
        normalized = np.zeros_like(eeg_data)
        for i in range(eeg_data.shape[0]):
            normalized[i] = normalize_eeg(eeg_data[i])
        return normalized
    else:
        raise ValueError(f"Unsupported EEG data shape: {eeg_data.shape}")


def preprocess_eeg(
    eeg_data: np.ndarray, 
    fs: float = 250, 
    lowcut: float = 5, 
    highcut: float = 40, 
    window_size: float = 1.0, 
    window_shift: float = 0.1
) -> List[np.ndarray]:
    """
    Apply full preprocessing pipeline to EEG data:
    1. Bandpass filtering
    2. Z-score normalization
    3. Segmentation
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        lowcut: Lower frequency bound in Hz
        highcut: Upper frequency bound in Hz
        window_size: Window size in seconds
        window_shift: Window shift in seconds
        
    Returns:
        List of preprocessed EEG segments
    """
    # Apply bandpass filter
    filtered_data = bandpass_filter(eeg_data, lowcut, highcut, fs)
    
    # Apply normalization
    normalized_data = normalize_eeg(filtered_data)
    
    # Segment data
    segments = segment_eeg(normalized_data, fs, window_size, window_shift)
    
    return segments


def load_and_preprocess_from_csv(
    csv_path: str,
    selected_classes: List[str] = None,
    channels: List[str] = None,
    fs: float = 250
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load EEG data from CSV file and preprocess it.
    
    Args:
        csv_path: Path to CSV file with EEG data
        selected_classes: List of class labels to include (e.g., ['left', 'right'])
        channels: List of channel names to include
        fs: Sampling frequency in Hz
        
    Returns:
        Tuple of (preprocessed_segments, labels)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter by selected classes if specified
    if selected_classes is not None:
        df = df[df['label'].isin(selected_classes)]
    
    # Keep only required columns
    if channels is None:
        # Use all EEG channels (assuming columns starting with 'EEG-')
        channels = [col for col in df.columns if col.startswith('EEG-')]
    
    # Group by patient and epoch to get trials
    trials = []
    labels = []
    
    for (patient, epoch), group in df.groupby(['patient', 'epoch']):
        # Sort by time to ensure correct order
        group = group.sort_values('time')
        
        # Extract channel data
        trial_data = group[channels].values.T  # Shape: (channels, samples)
        
        # Get label (assuming same label for the whole trial)
        label = group['label'].iloc[0]
        
        trials.append(trial_data)
        labels.append(label)
    
    # Convert string labels to numeric
    unique_labels = list(set(labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    numeric_labels = [label_map[lbl] for lbl in labels]
    
    # Preprocess trials
    preprocessed_segments = []
    segment_labels = []
    
    for trial, label in zip(trials, numeric_labels):
        segments = preprocess_eeg(trial, fs=fs)
        preprocessed_segments.extend(segments)
        segment_labels.extend([label] * len(segments))
    
    return preprocessed_segments, segment_labels


def split_dataset(
    data: List[np.ndarray], 
    labels: List[int],
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        data: List of EEG segments
        labels: List of labels
        train_ratio: Ratio of training set
        val_ratio: Ratio of validation set
        
    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    """
    # Convert to numpy arrays for easier manipulation
    data_array = np.array(data)
    labels_array = np.array(labels)
    
    # Compute indices for splits
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split the dataset
    train_data = data_array[train_indices].tolist()
    val_data = data_array[val_indices].tolist()
    test_data = data_array[test_indices].tolist()
    
    train_labels = labels_array[train_indices].tolist()
    val_labels = labels_array[val_indices].tolist()
    test_labels = labels_array[test_indices].tolist()
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def process_BCI_IV_dataset(
    npy_path: str,
    lowcut: float = 5,
    highcut: float = 40,
    fs: float = 250,
    window_size: float = 1.0,
    window_shift: float = 0.1
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Process BCI Competition IV dataset (specifically for the BCI_IV_2a_EEGclip file).
    
    Args:
        npy_path: Path to .npy file
        lowcut: Lower frequency bound in Hz
        highcut: Upper frequency bound in Hz
        fs: Sampling frequency in Hz
        window_size: Window size in seconds
        window_shift: Window shift in seconds
        
    Returns:
        Tuple of (preprocessed_segments, labels)
    """
    # Load data
    data = np.load(npy_path)
    
    # Assuming the data has shape (channels, samples) or similar
    # Apply preprocessing pipeline
    segments = preprocess_eeg(
        data, 
        fs=fs, 
        lowcut=lowcut, 
        highcut=highcut, 
        window_size=window_size, 
        window_shift=window_shift
    )
    
    # For this specific dataset, we're assuming binary classification
    # You may need to adjust this based on the actual labels in your dataset
    labels = [0] * len(segments)  # Placeholder, replace with actual labels if available
    
    return segments, labels 