"""
NPZ Converter Utility

This module provides functionality to convert EEG data from various formats to 
the standardized format expected by the MotorMind analysis pipeline.

It can be used both as a standalone script and as an imported module.
"""

import numpy as np
import sys
import os
import argparse
from pathlib import Path


def convert_npz_format(input_file, output_file):
    """
    Convert an NPZ file to the standardized format expected by MotorMind analysis.
    
    Parameters
    ----------
    input_file : str
        Path to the input NPZ file
    output_file : str
        Path to save the converted NPZ file
        
    Returns
    -------
    bool
        True if conversion was successful, False otherwise
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
            return False
        
        # Get the main data array
        main_data = data[data_key]
        
        # Determine channel count
        if len(main_data.shape) == 2:
            # Assuming format is (channels, samples)
            n_channels = main_data.shape[0]
        elif len(main_data.shape) == 3:
            # Assuming format is (segments, channels, samples)
            n_channels = main_data.shape[1]
        else:
            print(f"Unsupported data shape: {main_data.shape}")
            return False
        
        # Create a new file with the expected keys
        np.savez(
            output_file,
            data=main_data,
            fs=data['fs'] if 'fs' in data else 250,  # Default to 250 Hz if not present
            channels=data['channels'] if 'channels' in data else [f"Ch{i+1}" for i in range(n_channels)]
        )
        
        print(f"Converted file saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def main():
    """
    Main function when the module is run as a script.
    Parses command line arguments and performs the conversion.
    """
    parser = argparse.ArgumentParser(description='Convert NPZ files to MotorMind format')
    parser.add_argument('input_file', help='Path to the input NPZ file')
    parser.add_argument('output_file', help='Path to save the converted NPZ file')
    
    args = parser.parse_args()
    
    success = convert_npz_format(args.input_file, args.output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 