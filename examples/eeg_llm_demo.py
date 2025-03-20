#!/usr/bin/env python
"""
MotorMind EEG-LLM Demo

This script demonstrates how to use the MotorMind system to analyze EEG data using
Large Language Models with Supabase integration.

Requirements:
- numpy
- scipy
- mne (for EEG data handling)
- supabase
- requests
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.preprocessing.features import extract_all_features, features_to_text
from ml.data.preprocessing.eeg_preprocessing import preprocess_eeg, normalize_eeg, bandpass_filter
from ml.models.llm_wrapper import LLMWrapper, SupabaseLLMIntegration
from backend.database.supabase import SupabaseClient, create_database_schema


def load_eeg_data(file_path, format='numpy'):
    """
    Load EEG data from file.
    
    Args:
        file_path: Path to EEG data file
        format: File format ('numpy', 'edf', 'gdf')
        
    Returns:
        EEG data as numpy array, sampling rate, and channel names
    """
    if format == 'numpy':
        # Assuming .npz file with 'data', 'fs', and 'channels' keys
        data = np.load(file_path)
        return data['data'], data['fs'], data['channels']
    
    elif format in ['edf', 'gdf']:
        try:
            import mne
            raw = mne.io.read_raw(file_path, preload=True)
            data = raw.get_data()
            fs = raw.info['sfreq']
            channels = raw.ch_names
            return data, fs, channels
        except ImportError:
            print("MNE Python package required for EDF/GDF loading. Install with: pip install mne")
            sys.exit(1)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_dummy_eeg_data(duration=10, fs=250, n_channels=8):
    """
    Create dummy EEG data for demonstration purposes.
    
    Args:
        duration: Duration in seconds
        fs: Sampling frequency in Hz
        n_channels: Number of channels
        
    Returns:
        EEG data, sampling frequency, channel names
    """
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs
    
    # Create channel names
    standard_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    channels = standard_channels[:n_channels]
    
    # Create data with some alpha oscillations (8-12 Hz)
    data = np.zeros((n_channels, n_samples))
    
    # Alpha oscillations in posterior channels
    alpha_freq = 10  # Hz
    alpha_amp = 10  # µV
    posterior_idx = [i for i, ch in enumerate(channels) if ch[0] in 'PO']
    if posterior_idx:
        for idx in posterior_idx:
            data[idx] = alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
    
    # Beta oscillations in central channels
    beta_freq = 20  # Hz
    beta_amp = 5  # µV
    central_idx = [i for i, ch in enumerate(channels) if ch[0] == 'C']
    if central_idx:
        for idx in central_idx:
            data[idx] = beta_amp * np.sin(2 * np.pi * beta_freq * time)
    
    # Add some random noise
    noise_level = 2  # µV
    data += noise_level * np.random.randn(*data.shape)
    
    # Add motor imagery pattern (desynchronization in contralateral sensorimotor cortex)
    # Simulate right hand movement imagery (C3 desynchronization)
    c3_idx = channels.index('C3') if 'C3' in channels else None
    if c3_idx is not None:
        # Reduce amplitude in the middle of the recording
        mid_start = int(n_samples * 0.4)
        mid_end = int(n_samples * 0.6)
        data[c3_idx, mid_start:mid_end] *= 0.5
    
    return data, fs, channels


def process_eeg_data(eeg_data, fs, channels, window_size=2.0, overlap=0.5):
    """
    Process EEG data to extract features.
    
    Args:
        eeg_data: EEG data as numpy array (channels x samples)
        fs: Sampling frequency in Hz
        channels: Channel names
        window_size: Window size in seconds
        overlap: Window overlap ratio (0-1)
        
    Returns:
        List of features for each window
    """
    # First apply preprocessing steps from our new module
    # 1. Bandpass filter between 5-40 Hz (common for motor imagery)
    filtered_data = bandpass_filter(eeg_data, lowcut=5, highcut=40, fs=fs)
    
    # 2. Normalize the data
    normalized_data = normalize_eeg(filtered_data)
    
    # 3. Use the existing feature extraction for further processing
    window_samples = int(window_size * fs)
    features = extract_all_features(
        eeg_data=normalized_data,  # Use preprocessed data
        fs=fs,
        channel_names=channels,
        window_size=window_samples,
        overlap=overlap
    )
    
    return features


def setup_supabase_project(url, key, dataset_name="Motor Imagery Demo", user_id=None):
    """
    Set up Supabase project with necessary tables and a demo dataset.
    
    Args:
        url: Supabase project URL
        key: Supabase API key
        dataset_name: Name for the demo dataset
        user_id: User ID (if None, use authenticated user)
        
    Returns:
        Dictionary with dataset and recording IDs
    """
    # Create database schema
    schema_result = create_database_schema(url, key)
    print(f"Schema creation result: {schema_result}")
    
    # Initialize client
    supabase_client = SupabaseClient(url, key, debug=True)
    
    # Create demo dataset
    dataset_result = supabase_client.create_dataset(
        name=dataset_name,
        description="Demo dataset for EEG-LLM integration",
        metadata={"type": "motor_imagery", "source": "demo"},
        is_public=True,
        user_id=user_id
    )
    
    if not dataset_result.get("success"):
        print(f"Error creating dataset: {dataset_result}")
        return None
    
    dataset_id = dataset_result["data"]["id"]
    print(f"Created dataset with ID: {dataset_id}")
    
    # Create recording
    recording_result = supabase_client.create_recording(
        dataset_id=dataset_id,
        recording_date=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        duration=10,
        channels=8,
        sampling_rate=250,
        metadata={"task": "right_hand_motor_imagery"},
        user_id=user_id
    )
    
    if not recording_result.get("success"):
        print(f"Error creating recording: {recording_result}")
        return None
    
    recording_id = recording_result["data"]["id"]
    print(f"Created recording with ID: {recording_id}")
    
    return {
        "dataset_id": dataset_id,
        "recording_id": recording_id
    }


def store_eeg_features(supabase_client, recording_id, features):
    """
    Store extracted EEG features in Supabase.
    
    Args:
        supabase_client: SupabaseClient instance
        recording_id: Recording ID
        features: List of feature dictionaries
        
    Returns:
        Result of the operation
    """
    result = supabase_client.store_eeg_features(recording_id, features)
    print(f"Stored {result.get('count', 0)} feature sets")
    return result


def analyze_with_llm(
    feature_text,
    api_key=None,
    api_base="https://api.openai.com/v1",
    model_name="gpt-4",
    task="motor_imagery_classification"
):
    """
    Analyze EEG features using an LLM.
    
    Args:
        feature_text: Text representation of EEG features
        api_key: API key for LLM service
        api_base: API base URL
        model_name: LLM model name
        task: Analysis task
        
    Returns:
        Analysis results
    """
    llm = LLMWrapper(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1
    )
    
    result = llm.analyze_eeg(
        feature_text=feature_text,
        task=task,
        use_tree_of_thought=True
    )
    
    return result


def analyze_with_supabase_integration(
    supabase_url,
    supabase_key,
    recording_id,
    api_key=None,
    api_base="https://api.openai.com/v1",
    model_name="gpt-4",
    task="motor_imagery_classification"
):
    """
    Analyze EEG data using Supabase integration.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        recording_id: Recording ID
        api_key: API key for LLM service
        api_base: API base URL
        model_name: LLM model name
        task: Analysis task
        
    Returns:
        Analysis results
    """
    llm = LLMWrapper(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1
    )
    
    integration = SupabaseLLMIntegration(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        llm_wrapper=llm
    )
    
    result = integration.analyze_and_store(
        recording_id=recording_id,
        task=task,
        use_tree_of_thought=True
    )
    
    return result


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="MotorMind EEG-LLM Demo")
    parser.add_argument("--use-supabase", action="store_true", help="Use Supabase integration")
    parser.add_argument("--supabase-url", help="Supabase project URL")
    parser.add_argument("--supabase-key", help="Supabase API key")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4", help="LLM model name")
    parser.add_argument("--eeg-file", help="Path to EEG data file")
    parser.add_argument("--eeg-format", default="numpy", help="EEG file format (numpy, edf, gdf)")
    parser.add_argument("--output", default="results.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Set API key from arguments or environment
    api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("Warning: No LLM API key provided. Set --llm-api-key or LLM_API_KEY environment variable.")
    
    # Load or generate EEG data
    if args.eeg_file:
        print(f"Loading EEG data from {args.eeg_file}")
        eeg_data, fs, channels = load_eeg_data(args.eeg_file, args.eeg_format)
    else:
        print("Generating dummy EEG data")
        eeg_data, fs, channels = create_dummy_eeg_data()
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {channels}")
    
    # Extract features
    features = process_eeg_data(eeg_data, fs, channels)
    print(f"Extracted features for {len(features)} windows")
    
    # Convert first window features to text for demonstration
    feature_text = features_to_text(
        features[0],
        task_context="Motor imagery task - right hand movement"
    )
    
    results = {}
    
    if args.use_supabase:
        if not args.supabase_url or not args.supabase_key:
            print("Error: Supabase URL and key are required for Supabase integration")
            sys.exit(1)
        
        print("Setting up Supabase project")
        project_info = setup_supabase_project(args.supabase_url, args.supabase_key)
        
        if not project_info:
            print("Error setting up Supabase project")
            sys.exit(1)
        
        # Store features
        supabase_client = SupabaseClient(args.supabase_url, args.supabase_key, debug=True)
        store_eeg_features(supabase_client, project_info["recording_id"], features)
        
        # Analyze with Supabase integration
        print("Analyzing with Supabase integration")
        results = analyze_with_supabase_integration(
            supabase_url=args.supabase_url,
            supabase_key=args.supabase_key,
            recording_id=project_info["recording_id"],
            api_key=api_key,
            api_base=args.llm_api_base,
            model_name=args.llm_model,
            task="motor_imagery_classification"
        )
        
    else:
        # Direct analysis without Supabase
        print("Analyzing with LLM")
        analysis = analyze_with_llm(
            feature_text=feature_text,
            api_key=api_key,
            api_base=args.llm_api_base,
            model_name=args.llm_model,
            task="motor_imagery_classification"
        )
        
        results = {
            "success": True,
            "results": [analysis],
            "count": 1
        }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print classification and confidence
    if results.get("success") and results.get("results"):
        for i, result in enumerate(results["results"]):
            print(f"\nWindow {i+1} Analysis:")
            print(f"Classification: {result.get('classification')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Reasoning excerpt: {result.get('reasoning')[:500]}...")


if __name__ == "__main__":
    main() 