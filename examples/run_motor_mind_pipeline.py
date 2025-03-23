#!/usr/bin/env python

"""
MotorMind Full Pipeline Runner

This script demonstrates how to run the complete MotorMind pipeline:
1. Train an autoencoder tokenizer on EEG data
2. Train the MotorMind system with the autoencoder tokenizer
3. Perform inference with Gemini + RAG
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import time

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import numpy as np
        import tensorflow as tf
        from supabase import create_client, Client
        import google.generativeai as genai
        print("All dependencies are correctly installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required dependencies with:")
        print("pip install -r requirements.txt")
        return False


def create_directories():
    """Create required directories for models and visualizations"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    print("Created required directories.")


def load_credentials():
    """Load Supabase and Gemini credentials from environment or files"""
    credentials = {}
    
    # Try to load Supabase credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url and os.path.exists("supabase_url.txt"):
        with open("supabase_url.txt", "r") as f:
            supabase_url = f.read().strip()
    
    if not supabase_key and os.path.exists("supabase_key.txt"):
        with open("supabase_key.txt", "r") as f:
            supabase_key = f.read().strip()
    
    credentials["supabase_url"] = supabase_url
    credentials["supabase_key"] = supabase_key
    
    # Try to load Gemini credentials
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key and os.path.exists("gemini_api_key.txt"):
        with open("gemini_api_key.txt", "r") as f:
            gemini_api_key = f.read().strip()
    
    credentials["gemini_api_key"] = gemini_api_key
    
    # Check if credentials are available
    if not supabase_url or not supabase_key:
        print("WARNING: Supabase credentials not found. Vector storage will not work.")
    
    if not gemini_api_key:
        print("WARNING: Gemini API key not found. Inference will not work.")
    
    return credentials


def train_autoencoder(args, credentials):
    """Train the autoencoder tokenizer"""
    print("\n===== AUTOENCODER TRAINING =====")
    
    cmd = [
        sys.executable, "examples/train_autoencoder.py",
        "--eeg-file", args.eeg_file if args.eeg_file else "BCI_IV_2a_EEGclip (2).npy",
        "--eeg-format", args.eeg_format,
        "--fs", str(args.fs),
        "--latent-dim", str(args.latent_dim),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--model-output", args.model_output,
        "--visualize",
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print("Error training autoencoder:")
        print(process.stderr)
        return False
    
    print(process.stdout)
    print("\nAutoencoder training completed successfully!")
    return True


def train_motormind(args, credentials):
    """Train the MotorMind system with the autoencoder tokenizer"""
    print("\n===== MOTORMIND TRAINING =====")
    
    cmd = [
        sys.executable, "examples/eeg_rag_demo.py",
        "--eeg-file", args.eeg_file if args.eeg_file else "BCI_IV_2a_EEGclip (2).npy",
        "--eeg-format", args.eeg_format,
        "--fs", str(args.fs),
        "--tokenizer", "autoencoder",
        "--autoencoder-model-path", args.model_output,
        "--skip-inference",
    ]
    
    if credentials["supabase_url"]:
        cmd.extend(["--supabase-url", credentials["supabase_url"]])
    
    if credentials["supabase_key"]:
        cmd.extend(["--supabase-key", credentials["supabase_key"]])
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print("Error training MotorMind system:")
        print(process.stderr)
        return False
    
    print(process.stdout)
    print("\nMotorMind training completed successfully!")
    return True


def run_inference(args, credentials):
    """Run inference with the trained MotorMind system"""
    print("\n===== MOTORMIND INFERENCE =====")
    
    cmd = [
        sys.executable, "examples/eeg_rag_demo.py",
        "--eeg-file", args.eeg_file if args.eeg_file else "BCI_IV_2a_EEGclip (2).npy",
        "--eeg-format", args.eeg_format,
        "--fs", str(args.fs),
        "--inference-tokenizer", "autoencoder",
        "--autoencoder-model-path", args.model_output,
        "--task", args.task,
        "--skip-training",
    ]
    
    if credentials["supabase_url"]:
        cmd.extend(["--supabase-url", credentials["supabase_url"]])
    
    if credentials["supabase_key"]:
        cmd.extend(["--supabase-key", credentials["supabase_key"]])
    
    if credentials["gemini_api_key"]:
        cmd.extend(["--gemini-api-key", credentials["gemini_api_key"]])
    
    if args.disable_rag:
        cmd.append("--disable-rag")
    
    if args.disable_tot:
        cmd.append("--disable-tot")
    
    if args.debug:
        cmd.append("--debug")
    
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print("Error running inference:")
        print(process.stderr)
        return False
    
    print(process.stdout)
    print("\nMotorMind inference completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the complete MotorMind pipeline")
    
    # Data parameters
    parser.add_argument("--eeg-file", help="Path to EEG data file")
    parser.add_argument("--eeg-format", default="numpy", help="EEG file format (numpy, edf, gdf)")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency in Hz")
    
    # Autoencoder parameters
    parser.add_argument("--latent-dim", type=int, default=64, help="Dimension of the latent space")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--model-output", default="models/autoencoder.keras", help="Path to save the model")
    
    # Inference parameters
    parser.add_argument("--task", choices=["motor_imagery_classification", "thought_to_text", "abnormality_detection"], 
                       default="motor_imagery_classification", help="Analysis task to perform")
    parser.add_argument("--disable-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--disable-tot", action="store_true", help="Disable tree-of-thought reasoning")
    
    # Pipeline control
    parser.add_argument("--skip-autoencoder-training", action="store_true", help="Skip autoencoder training")
    parser.add_argument("--skip-tokenization-training", action="store_true", help="Skip tokenization training")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference phase")
    
    # Debug parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create required directories
    create_directories()
    
    # Load credentials
    credentials = load_credentials()
    
    # Start timing
    start_time = time.time()
    
    # Train autoencoder if not skipped
    if not args.skip_autoencoder_training:
        autoencoder_success = train_autoencoder(args, credentials)
        if not autoencoder_success:
            print("Autoencoder training failed. Stopping pipeline.")
            return
    else:
        print("\nSkipping autoencoder training as requested.")
    
    # Train MotorMind system if not skipped
    if not args.skip_tokenization_training:
        if not os.path.exists(args.model_output) and args.skip_autoencoder_training:
            print(f"ERROR: Autoencoder model not found at {args.model_output}")
            print("Cannot proceed with tokenization training without an autoencoder model.")
            return
        
        tokenization_success = train_motormind(args, credentials)
        if not tokenization_success:
            print("MotorMind tokenization training failed. Stopping pipeline.")
            return
    else:
        print("\nSkipping tokenization training as requested.")
    
    # Run inference if not skipped
    if not args.skip_inference:
        if not os.path.exists(args.model_output) and args.skip_autoencoder_training:
            print(f"ERROR: Autoencoder model not found at {args.model_output}")
            print("Cannot proceed with inference without an autoencoder model.")
            return
        
        if not credentials["gemini_api_key"]:
            print("ERROR: Gemini API key not found. Cannot proceed with inference.")
            return
        
        inference_success = run_inference(args, credentials)
        if not inference_success:
            print("MotorMind inference failed.")
            return
    else:
        print("\nSkipping inference as requested.")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print("\n===== PIPELINE SUMMARY =====")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("All requested steps completed successfully!")
    print("\nTo run individual components:")
    print(f"  - Train autoencoder: python examples/train_autoencoder.py --model-output {args.model_output}")
    print(f"  - Tokenize EEG data: python examples/eeg_rag_demo.py --tokenizer autoencoder --autoencoder-model-path {args.model_output} --skip-inference")
    print(f"  - Run inference: python examples/eeg_rag_demo.py --inference-tokenizer autoencoder --autoencoder-model-path {args.model_output} --skip-training")


if __name__ == "__main__":
    main() 