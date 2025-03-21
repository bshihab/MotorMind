#!/usr/bin/env python
"""
NPZ to NPY Converter Script

This script extracts the main data array from an NPZ file and saves it as a simple NPY file
for use with the MotorMind analysis pipeline.

Usage:
    python convert_npz.py input_file.npz output_file.npy

Example:
    python convert_npz.py processed_eeg_data.npz processed_data.npy
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def convert_npz_to_npy(input_file, output_file):
    """
    Extract the main data array from an NPZ file and save as a simple NPY file.
    
    Args:
        input_file: Path to the input NPZ file
        output_file: Path to save the output NPY file
        
    Returns:
        Tuple of (success, sampling_rate, n_channels)
    """
    try:
        # Load the processed data
        data = np.load(input_file, allow_pickle=True)
        
        # Print the keys to see what's available
        print("Keys in the file:", list(data.keys()))
        
        # Determine which key to use for the main data
        data_keys = ['preprocessed_data', 'data', 'raw_data', 'eeg_data', 'segments']
        data_key = None
        
        for key in data_keys:
            if key in data:
                data_key = key
                break
                
        if data_key is None:
            print("Could not find a valid data key in the NPZ file.")
            print("Available keys:", list(data.keys()))
            print("Looking for one of:", data_keys)
            return False, None, None
        
        # Get the main data array
        main_data = data[data_key]
        
        # Extract metadata if available
        sampling_rate = data['fs'] if 'fs' in data else 250
        if 'channels' in data:
            channels = data['channels']
            n_channels = len(channels)
        else:
            # Determine channel count from data shape
            if len(main_data.shape) == 2:
                # Assuming format is (channels, samples)
                n_channels = main_data.shape[0]
            elif len(main_data.shape) == 3:
                # Assuming format is (segments, channels, samples)
                n_channels = main_data.shape[1]
            else:
                n_channels = 0
        
        # Save the main data as a simple .npy file
        np.save(output_file, main_data)
        
        # Save metadata to a text file
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Converted from: {input_file}\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Number of channels: {n_channels}\n")
            f.write(f"Data shape: {main_data.shape}\n")
            f.write(f"Source data key: {data_key}\n")
        
        print(f"Data saved to {output_file}")
        print(f"Metadata saved to {metadata_file}")
        return True, sampling_rate, n_channels
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False, None, None


def main():
    """
    Main function for command-line usage.
    """
    # Simple argument validation
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_file.npz output_file.npy")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Ensure output file has .npy extension
    if not output_file.lower().endswith('.npy'):
        output_file += '.npy'
        print(f"Adding .npy extension to output file: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Convert the file
    print(f"Converting '{input_file}' to '{output_file}'...")
    success, sampling_rate, n_channels = convert_npz_to_npy(input_file, output_file)
    
    if success:
        print("Conversion successful!")
        print("\nYou can now use this file with the EEG-LLM demo:")
        print(f"python examples/eeg_llm_demo.py --eeg-file {output_file} --eeg-format simple_numpy --sampling-rate {sampling_rate} --channels {n_channels}")
    else:
        print("Conversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 