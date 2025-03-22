"""
EEG Data Loader

This module provides functions for loading EEG data from various file formats,
and generating dummy data for testing purposes.
"""

import os
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import random


def load_eeg_data(
    file_path: str,
    file_format: str = "numpy"
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load EEG data from a file.
    
    Args:
        file_path: Path to the EEG data file
        file_format: Format of the file ('numpy', 'edf', 'gdf')
        
    Returns:
        Tuple of (eeg_data, sampling_frequency, channel_names)
    """
    # Default values
    sampling_frequency = 250.0  # Hz
    
    try:
        if file_format.lower() == "numpy":
            # Load numpy file
            if file_path.endswith('.npz'):
                with np.load(file_path) as data:
                    # Try to find the EEG data array
                    eeg_data = None
                    for key in data.files:
                        if 'data' in key.lower() or 'eeg' in key.lower():
                            eeg_data = data[key]
                            break
                    
                    # If no suitable key is found, use the first one
                    if eeg_data is None and len(data.files) > 0:
                        eeg_data = data[data.files[0]]
                    
                    # Try to find sampling frequency and channel names
                    for key in data.files:
                        if 'fs' in key.lower() or 'samplerate' in key.lower() or 'sample_rate' in key.lower():
                            sampling_frequency = float(data[key])
                        
                        if 'channel' in key.lower() or 'ch_names' in key.lower():
                            channel_names = data[key].tolist() if isinstance(data[key], np.ndarray) else data[key]
            else:
                # Regular .npy file
                eeg_data = np.load(file_path)
            
            # Generate default channel names if not found
            if 'channel_names' not in locals() or channel_names is None:
                channel_names = [f"Ch{i+1}" for i in range(eeg_data.shape[0])]
            
            return eeg_data, sampling_frequency, channel_names
        
        elif file_format.lower() == "edf":
            # Use MNE to load EDF file
            try:
                import mne
                raw = mne.io.read_raw_edf(file_path, preload=True)
                eeg_data = raw.get_data()
                sampling_frequency = raw.info['sfreq']
                channel_names = raw.ch_names
                
                return eeg_data, sampling_frequency, channel_names
            except ImportError:
                print("MNE package not found. Install with: pip install mne")
                # Fallback to dummy data
                return create_dummy_eeg_data()
        
        elif file_format.lower() == "gdf":
            # Use MNE to load GDF file
            try:
                import mne
                raw = mne.io.read_raw_gdf(file_path, preload=True)
                eeg_data = raw.get_data()
                sampling_frequency = raw.info['sfreq']
                channel_names = raw.ch_names
                
                return eeg_data, sampling_frequency, channel_names
            except ImportError:
                print("MNE package not found. Install with: pip install mne")
                # Fallback to dummy data
                return create_dummy_eeg_data()
        
        else:
            print(f"Unsupported file format: {file_format}")
            # Fallback to dummy data
            return create_dummy_eeg_data()
    
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        # Fallback to dummy data
        return create_dummy_eeg_data()


def create_dummy_eeg_data(
    num_channels: int = 16,
    duration_seconds: float = 10.0,
    sampling_frequency: float = 250.0,
    has_motor_imagery: bool = True,
    motor_imagery_type: str = "right_hand"
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Create dummy EEG data for testing.
    
    Args:
        num_channels: Number of channels in the data
        duration_seconds: Duration of the data in seconds
        sampling_frequency: Sampling frequency in Hz
        has_motor_imagery: Whether to include simulated motor imagery patterns
        motor_imagery_type: Type of motor imagery to simulate ('right_hand', 'left_hand', 'feet')
        
    Returns:
        Tuple of (eeg_data, sampling_frequency, channel_names)
    """
    # Calculate number of samples
    num_samples = int(duration_seconds * sampling_frequency)
    
    # Generate time array
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Create standard 10-20 channel names if num_channels is compatible
    if num_channels in [19, 21]:
        channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
        ]
        if num_channels == 21:
            channel_names.extend(['A1', 'A2'])
    else:
        # For other numbers of channels, create generic names
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    # Initialize the EEG data array
    eeg_data = np.zeros((num_channels, num_samples))
    
    # Generate base signal for each channel (random noise + alpha oscillations)
    for i in range(num_channels):
        # Add pink noise
        noise = generate_pink_noise(num_samples)
        
        # Add alpha oscillations (8-13 Hz)
        alpha = 2.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha rhythm
        
        # Add some theta (4-8 Hz)
        theta = 1.0 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta rhythm
        
        # Add some beta (13-30 Hz)
        beta = 0.5 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta rhythm
        
        # Combine signals
        eeg_data[i, :] = noise + alpha + theta + beta
    
    # Add motor imagery patterns if requested
    if has_motor_imagery:
        # Add event-related desynchronization (ERD) in the mu rhythm (8-13 Hz)
        # ERD is characterized by a decrease in power in the mu band
        
        # Define which channels will show the ERD effect based on motor imagery type
        if motor_imagery_type == "right_hand":
            # Left hemisphere (C3) should show stronger ERD for right hand imagery
            target_channels = ["C3"]
        elif motor_imagery_type == "left_hand":
            # Right hemisphere (C4) should show stronger ERD for left hand imagery
            target_channels = ["C4"]
        elif motor_imagery_type == "feet":
            # Central area (Cz) should show ERD for feet imagery
            target_channels = ["Cz"]
        else:
            # Default to bilateral
            target_channels = ["C3", "C4"]
        
        # Create a time window for the ERD (e.g., from 2-8 seconds)
        erd_start = int(2 * sampling_frequency)
        erd_end = int(8 * sampling_frequency)
        
        # Apply ERD to target channels
        for channel in target_channels:
            if channel in channel_names:
                idx = channel_names.index(channel)
                
                # Create a modulation signal (Gaussian envelope)
                modulation = np.ones(num_samples)
                modulation[erd_start:erd_end] = 0.5  # 50% reduction in mu power
                
                # Apply the modulation to the alpha rhythm
                eeg_data[idx, :] = eeg_data[idx, :] * modulation
    
    return eeg_data, sampling_frequency, channel_names


def generate_pink_noise(num_samples: int) -> np.ndarray:
    """
    Generate pink noise (1/f noise).
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Pink noise array
    """
    # Generate white noise
    white_noise = np.random.normal(0, 1, num_samples)
    
    # Convert to frequency domain
    noise_fft = np.fft.rfft(white_noise)
    
    # Create pink noise by applying 1/f filter
    f = np.fft.rfftfreq(num_samples)
    f[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(f)
    
    # Apply filter
    pink_noise_fft = noise_fft * pink_filter
    
    # Convert back to time domain
    pink_noise = np.fft.irfft(pink_noise_fft)
    
    # Normalize
    pink_noise = pink_noise / np.std(pink_noise)
    
    return pink_noise 