#!/usr/bin/env python
"""
EEG Preprocessing Example Script

This script demonstrates how to preprocess EEG data using the MotorMind preprocessing module.
It can work with both CSV and .npy formats of EEG data.

Example usage:
    python preprocess_eeg_data.py --input BCI_IV_2a_EEGclip.npy --output preprocessed_eeg.npy --fs 250
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.preprocessing.eeg_preprocessing import (
    process_BCI_IV_dataset,
    load_and_preprocess_from_csv,
    split_dataset
)


def main():
    """Main function to run EEG preprocessing example."""
    parser = argparse.ArgumentParser(description="MotorMind EEG Preprocessing Example")
    parser.add_argument("--input", required=True, help="Path to input EEG data file (.npy or .csv)")
    parser.add_argument("--output", default="preprocessed_eeg.npy", help="Path to output preprocessed data file")
    parser.add_argument("--fs", type=float, default=250, help="Sampling frequency in Hz")
    parser.add_argument("--lowcut", type=float, default=5, help="Lowcut frequency for bandpass filter")
    parser.add_argument("--highcut", type=float, default=40, help="Highcut frequency for bandpass filter")
    parser.add_argument("--window-size", type=float, default=1.0, help="Window size in seconds")
    parser.add_argument("--window-shift", type=float, default=0.1, help="Window shift in seconds")
    parser.add_argument("--selected-classes", nargs='+', help="Classes to include (for CSV only, e.g., 'left right')")
    
    args = parser.parse_args()
    
    # Determine file format
    file_ext = os.path.splitext(args.input)[1].lower()
    
    print(f"Processing EEG data from {args.input}...")
    
    if file_ext == '.npy':
        # Process .npy file (BCI Competition format)
        segments, labels = process_BCI_IV_dataset(
            npy_path=args.input,
            lowcut=args.lowcut,
            highcut=args.highcut,
            fs=args.fs,
            window_size=args.window_size,
            window_shift=args.window_shift
        )
        print(f"Processed {len(segments)} segments from .npy file")
        
    elif file_ext == '.csv':
        # Process CSV file
        segments, labels = load_and_preprocess_from_csv(
            csv_path=args.input,
            selected_classes=args.selected_classes,
            fs=args.fs
        )
        print(f"Processed {len(segments)} segments from CSV file")
        
    else:
        print(f"Unsupported file format: {file_ext}")
        return
    
    # Split into train/val/test sets
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
        segments, labels, train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Dataset split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test segments")
    
    # Convert lists to numpy arrays for easier saving
    train_data_array = np.array(train_data)
    val_data_array = np.array(val_data)
    test_data_array = np.array(test_data)
    
    # Determine base output path without extension
    base_output = os.path.splitext(args.output)[0]
    
    # Save each partition as a separate .npy file
    np.save(f"{base_output}_train_data.npy", train_data_array)
    np.save(f"{base_output}_train_labels.npy", np.array(train_labels))
    np.save(f"{base_output}_val_data.npy", val_data_array)
    np.save(f"{base_output}_val_labels.npy", np.array(val_labels))
    np.save(f"{base_output}_test_data.npy", test_data_array)
    np.save(f"{base_output}_test_labels.npy", np.array(test_labels))
    
    # Save preprocessed data - main output is the training data
    np.save(args.output, train_data_array)
    
    # Save metadata to a text file
    metadata_file = f"{base_output}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Sampling rate: {args.fs} Hz\n")
        f.write(f"Data splits:\n")
        f.write(f"  Train: {train_data_array.shape}, labels: {len(train_labels)}\n")
        f.write(f"  Validation: {val_data_array.shape}, labels: {len(val_labels)}\n")
        f.write(f"  Test: {test_data_array.shape}, labels: {len(test_labels)}\n")
        if len(train_data) > 0:
            f.write(f"Example segment shape: {train_data[0].shape}\n")
    
    print(f"Preprocessed data saved to {args.output}")
    print(f"Data splits saved to:")
    print(f"  {base_output}_train_data.npy")
    print(f"  {base_output}_val_data.npy")
    print(f"  {base_output}_test_data.npy")
    print(f"Metadata saved to {metadata_file}")
    
    # Print example segment shape for reference
    if len(train_data) > 0:
        print(f"Example segment shape: {train_data[0].shape}")
    
    print(f"\nIMPORTANT: When using these files with eeg_llm_demo.py, use these parameters:")
    print(f"  --eeg-format simple_numpy --sampling-rate {args.fs} --channels {train_data[0].shape[0] if len(train_data) > 0 else 'N/A'}")


if __name__ == "__main__":
    main() 