#!/usr/bin/env python
"""
BCI IV 2a Dataset Processing Utility

This script provides a way to process the BCI_IV_2a_EEGclip.npy file format
and prepare it for use with the MotorMind system.

Example usage:
    python process_bci_iv_file.py --input "BCI_IV_2a_EEGclip (2).npy" --output processed_eeg_data.npz
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.preprocessing.eeg_preprocessing import (
    bandpass_filter, 
    normalize_eeg, 
    preprocess_eeg, 
    segment_eeg
)
from ml.data.preprocessing.features import extract_all_features, features_to_text


def load_bci_iv_data(file_path):
    """
    Load BCI IV 2a dataset from .npy file.
    
    Args:
        file_path: Path to .npy file
        
    Returns:
        EEG data array, sampling rate, and channel names
    """
    try:
        data = np.load(file_path)
        print(f"Loaded data shape: {data.shape}")
        
        # The file has shape (22, 1500) - 22 channels, 1500 time points (best guess)
        # Assume sampling rate of 250 Hz (common for BCI datasets)
        fs = 250
        
        # Create default channel names if not available
        channels = [f"Ch{i+1}" for i in range(data.shape[0])]
        
        return data, fs, channels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def visualize_eeg(data, fs, channels=None, title="EEG Signal", n_channels=5):
    """
    Visualize EEG data.
    
    Args:
        data: EEG data array (channels x samples)
        fs: Sampling frequency
        channels: Channel names
        title: Plot title
        n_channels: Number of channels to plot
    """
    if channels is None:
        channels = [f"Ch{i+1}" for i in range(data.shape[0])]
    
    # Select subset of channels to plot
    n_channels = min(n_channels, data.shape[0])
    
    plt.figure(figsize=(15, 10))
    
    # Create time axis
    time = np.arange(data.shape[1]) / fs
    
    # Plot each channel
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time, data[i])
        plt.title(f"Channel {channels[i]}")
        plt.ylabel("Amplitude")
        if i == n_channels - 1:
            plt.xlabel("Time (s)")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def process_and_save(input_file, output_file, visualize=False):
    """
    Process BCI IV data and save preprocessed features.
    
    Args:
        input_file: Path to input .npy file
        output_file: Path to output .npz file
        visualize: Whether to visualize the data
    """
    # Load data
    eeg_data, fs, channels = load_bci_iv_data(input_file)
    
    if eeg_data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Visualize raw data if requested
    if visualize:
        visualize_eeg(eeg_data, fs, channels, title="Raw EEG Data")
    
    # Preprocess data
    filtered_data = bandpass_filter(eeg_data, lowcut=5, highcut=40, fs=fs)
    normalized_data = normalize_eeg(filtered_data)
    
    # Visualize preprocessed data if requested
    if visualize:
        visualize_eeg(normalized_data, fs, channels, title="Preprocessed EEG Data")
    
    # Segment data
    window_size = 1.0  # 1 second
    window_shift = 0.25  # 250 ms
    segments = segment_eeg(normalized_data, fs, window_size, window_shift)
    
    print(f"Created {len(segments)} segments")
    print(f"Segment shape: {segments[0].shape}")
    
    # Extract features
    features = []
    for segment in segments:
        segment_features = extract_all_features(
            eeg_data=segment, 
            fs=fs, 
            channel_names=channels,
            window_size=segment.shape[1],
            overlap=0  # No additional segmentation
        )
        # Since extract_all_features returns a list, we take the first item
        features.append(segment_features[0])
    
    print(f"Extracted features for {len(features)} segments")
    
    # Convert features to text representation for first example
    feature_text = features_to_text(
        features[0], 
        task_context="BCI IV 2a motor imagery dataset analysis"
    )
    
    print("Example feature text:")
    print(feature_text[:500] + "...\n")
    
    # Save preprocessed data
    np.savez(
        output_file,
        raw_data=eeg_data,
        preprocessed_data=normalized_data,
        segments=np.array(segments),
        features=features,
        fs=fs,
        channels=channels
    )
    
    print(f"Preprocessed data saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="BCI IV 2a Dataset Processing Utility")
    parser.add_argument("--input", required=True, help="Path to BCI_IV_2a_EEGclip.npy file")
    parser.add_argument("--output", default="processed_eeg_data.npz", help="Path to output file")
    parser.add_argument("--visualize", action="store_true", help="Visualize the data (requires matplotlib)")
    
    args = parser.parse_args()
    
    process_and_save(args.input, args.output, args.visualize)


if __name__ == "__main__":
    main() 