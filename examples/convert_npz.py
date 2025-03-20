#!/usr/bin/env python
"""
NPZ Converter Script

This script converts EEG data NPZ files to the standardized format 
expected by the MotorMind analysis pipeline.

Usage:
    python convert_npz.py input_file.npz output_file.npz

Example:
    python convert_npz.py processed_eeg_data.npz converted_data.npz
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.preprocessing.npz_converter import convert_npz_format


def main():
    """
    Main function for command-line usage.
    """
    # Simple argument validation
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_file.npz output_file.npz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Convert the file
    print(f"Converting '{input_file}' to '{output_file}'...")
    success = convert_npz_format(input_file, output_file)
    
    if success:
        print("Conversion successful!")
        print("\nYou can now use this file with the EEG-LLM demo:")
        print(f"python examples/eeg_llm_demo.py --eeg-file {output_file} --llm-api-key YOUR_API_KEY")
    else:
        print("Conversion failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 