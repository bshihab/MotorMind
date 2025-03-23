#!/usr/bin/env python
"""
MotorMind EEG-RAG Demo

This script demonstrates the full MotorMind pipeline using the new architecture:
1. Loading EEG data
2. Tokenizing the data using different algorithms
3. Storing tokens in the Supabase vector database
4. Performing inference with Gemini + RAG
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import MotorMind components
from eeg_acquisition.data_collection.eeg_loader import load_eeg_data, create_dummy_eeg_data
from tokenization.feature_domain.feature_tokenizer import FeatureTokenizer
from tokenization.frequency_domain.frequency_tokenizer import FrequencyTokenizer
from tokenization.autoencoder.autoencoder_tokenizer import AutoencoderTokenizer
from vector_store.database.supabase_vector_store import SupabaseVectorStore
from training.data_processing.eeg_tokenizer_pipeline import EEGTokenizerPipeline
from inference.llm.eeg_rag_engine import EEGRAGEngine
from inference.llm.gemini_client import GeminiClient

# Import Supabase client
from vector_store.database.supabase_client import setup_supabase_client as init_supabase


def setup_supabase_client(url=None, key=None):
    """
    Set up and return a Supabase client instance.
    
    Args:
        url: Supabase project URL (if None, read from env)
        key: Supabase API key (if None, read from env)
        
    Returns:
        Initialized SupabaseWrapper
    """
    # Use the helper function from supabase_client.py
    supabase = init_supabase(url, key, debug=True)
    
    if supabase is None:
        print("Error: Could not connect to Supabase")
        return None
    
    print("Supabase client initialized successfully")
    return supabase


def setup_gemini_client(api_key=None, model_name="gemini-pro"):
    """
    Set up and return a Gemini client instance.
    
    Args:
        api_key: API key for the Gemini service
        model_name: Name of the Gemini model to use
        
    Returns:
        Initialized Gemini client
    """
    # Use provided value or try to load from file
    try:
        # Initialize client
        gemini_client = GeminiClient(api_key=api_key, model_name=model_name)
        
        print(f"Gemini client initialized with model {model_name}")
        return gemini_client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        print("Make sure gemini_api_key.txt exists in the project root or set GEMINI_API_KEY environment variable")
        return None


def demo_training_phase(eeg_data, channel_names, fs, supabase_client, tokenizer_type="both", autoencoder_model_path=None, debug=False):
    """
    Demonstrate the training phase of the MotorMind pipeline.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        channel_names: List of channel names
        fs: Sampling frequency in Hz
        supabase_client: Supabase client instance
        tokenizer_type: Type of tokenizer to use ("feature", "frequency", "autoencoder", "both", or "all")
        autoencoder_model_path: Path to pre-trained autoencoder model (for "autoencoder" tokenizer type)
        debug: Enable debug output
        
    Returns:
        Dictionary with training results
    """
    print("\n===== TRAINING PHASE =====")
    print(f"Tokenizing EEG data with {tokenizer_type} tokenizer...")
    
    start_time = time.time()
    
    # Create vector store with debug enabled
    vector_store = SupabaseVectorStore(supabase_client, debug=True)
    
    # Set up tokenizer parameters
    tokenizer_params = {
        "fs": fs,
        "debug": debug
    }
    
    # Add autoencoder model path if provided
    if autoencoder_model_path:
        tokenizer_params["model_path"] = autoencoder_model_path
    
    # Set up tokenizers to use
    tokenizers_to_use = []
    if tokenizer_type == "feature" or tokenizer_type == "both" or tokenizer_type == "all":
        tokenizers_to_use.append("feature")
    if tokenizer_type == "frequency" or tokenizer_type == "both" or tokenizer_type == "all":
        tokenizers_to_use.append("frequency")
    if tokenizer_type == "autoencoder" or tokenizer_type == "all":
        if autoencoder_model_path:
            tokenizers_to_use.append("autoencoder")
        else:
            print("WARNING: Autoencoder tokenizer requested but no model path provided. Skipping autoencoder tokenization.")
    
    # Create tokenizer pipeline with debug enabled
    pipeline = EEGTokenizerPipeline(
        tokenizer_type=tokenizer_type,
        tokenizer_params=tokenizer_params,
        vector_store=vector_store,
        debug=True  # Enable debug to get more information
    )
    
    # Process EEG data
    results = pipeline.process_eeg_data(
        eeg_data=eeg_data,
        channel_names=channel_names
    )
    
    training_time = time.time() - start_time
    
    # Print results
    print("\nTokenization Results:")
    for tokenizer in results["tokenization"]:
        print(f"  {tokenizer}: {results['tokenization'][tokenizer]['count']} tokens generated")
    
    print(f"\nStorage Results:")
    print(f"  Success: {results['storage']['success']}")
    print(f"  Tokens stored: {results['storage']['count']}")
    
    # Print detailed storage errors if available
    if not results['storage']['success'] and 'tokenizer_results' in results['storage']:
        print("\nStorage Error Details:")
        for tokenizer_type, result in results['storage']['tokenizer_results'].items():
            if not result.get('success', False):
                print(f"  {tokenizer_type}: {result.get('error', 'Unknown error')}")
    
    print(f"\nTraining phase completed in {training_time:.2f} seconds")
    
    return results


def demo_inference_phase(eeg_data, channel_names, fs, supabase_client, gemini_client, 
                        tokenizer_type="frequency", autoencoder_model_path=None,
                        task="motor_imagery_classification", use_rag=True, debug=False):
    """
    Demonstrate the inference phase of the MotorMind pipeline.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        channel_names: List of channel names
        fs: Sampling frequency in Hz
        supabase_client: Supabase client instance
        gemini_client: Gemini client instance
        tokenizer_type: Type of tokenizer to use ("feature", "frequency", or "autoencoder")
        autoencoder_model_path: Path to pre-trained autoencoder model (for "autoencoder" tokenizer type)
        task: Analysis task to perform
        use_rag: Whether to use RAG
        debug: Enable debug output
        
    Returns:
        Dictionary with inference results
    """
    print("\n===== INFERENCE PHASE =====")
    print(f"Analyzing EEG data with {tokenizer_type} tokenizer...")
    
    start_time = time.time()
    
    # Create vector store if using RAG
    vector_store = SupabaseVectorStore(supabase_client) if use_rag else None
    
    # Set up tokenizer parameters
    tokenizer_params = {
        "fs": fs,
        "debug": debug
    }
    
    # Add model_path only for autoencoder tokenizer
    if tokenizer_type == "autoencoder" and autoencoder_model_path:
        tokenizer_params["model_path"] = autoencoder_model_path
    
    # Create RAG engine
    rag_engine = EEGRAGEngine(
        gemini_client=gemini_client,
        tokenizer_type=tokenizer_type,
        tokenizer_params=tokenizer_params,
        vector_store=vector_store,
        debug=debug
    )
    
    # Process query segment from EEG data
    # For demo purposes, use a segment from the middle of the data
    segment_duration = 2.0  # seconds
    segment_samples = int(segment_duration * fs)
    
    # Ensure we don't exceed the data length
    max_start = max(0, eeg_data.shape[1] - segment_samples - 1)
    start_idx = max_start // 2  # Take from the middle
    
    # Extract segment
    eeg_segment = eeg_data[:, start_idx:start_idx + segment_samples]
    
    # Run inference
    results = rag_engine.process_eeg_data(
        eeg_data=eeg_segment,
        channel_names=channel_names,
        task=task
    )
    
    inference_time = time.time() - start_time
    
    # Print results
    print("\nInference Results:")
    print(f"  Task: {task}")
    
    # Map the new result structure to our expected output
    inference_result = {
        'prediction': None,
        'confidence': None,
        'text': None,
        'evidence': []
    }
    
    # Extract interpretation from the new structure
    if 'interpretation' in results:
        interpretation = results['interpretation']
        
        if task == "motor_imagery_classification":
            inference_result['prediction'] = interpretation.get('class')
            inference_result['confidence'] = interpretation.get('confidence')
        elif task == "thought_to_text":
            inference_result['text'] = interpretation.get('text')
            inference_result['confidence'] = interpretation.get('confidence')
        elif task == "abnormality_detection":
            inference_result['abnormality_detected'] = interpretation.get('class') == 'abnormal'
            inference_result['abnormality_type'] = interpretation.get('abnormality_type', 'Unknown')
            inference_result['severity'] = interpretation.get('severity', 'Unknown')
            inference_result['confidence'] = interpretation.get('confidence')
    
    # Extract supporting evidence if RAG was used
    if use_rag and 'rag' in results and 'similar_examples' in results['rag']:
        inference_result['evidence'] = [
            {'text': example.get('token_text', ''), 'score': example.get('similarity', 0)}
            for example in results['rag']['similar_examples']
        ]
    
    # Print formatted results based on the task
    if task == "motor_imagery_classification":
        print(f"  Predicted class: {inference_result['prediction']}")
        confidence = inference_result['confidence']
        if confidence is not None:
            print(f"  Confidence: {confidence:.2f}")
        else:
            print("  Confidence: N/A")
    elif task == "thought_to_text":
        print(f"  Decoded thought: {inference_result['text']}")
        confidence = inference_result['confidence']
        if confidence is not None:
            print(f"  Confidence: {confidence:.2f}")
        else:
            print("  Confidence: N/A")
    elif task == "abnormality_detection":
        print(f"  Abnormality detected: {inference_result['abnormality_detected']}")
        if inference_result['abnormality_detected']:
            print(f"  Abnormality type: {inference_result['abnormality_type']}")
            print(f"  Severity: {inference_result['severity']}")
        confidence = inference_result['confidence']
        if confidence is not None:
            print(f"  Confidence: {confidence:.2f}")
        else:
            print("  Confidence: N/A")
    
    # Print supporting evidence if RAG was used
    if use_rag and inference_result['evidence']:
        print("\nSupporting Evidence:")
        for e in inference_result['evidence'][:3]:  # Show top 3 pieces of evidence
            print(f"  - {e['text']} (Score: {e['score']:.2f})")
    
    print(f"\nInference phase completed in {inference_time:.2f} seconds")
    
    return inference_result


def main():
    """Main function to run the MotorMind EEG-RAG Demo."""
    parser = argparse.ArgumentParser(description="MotorMind EEG-RAG Demo")
    
    parser.add_argument("--eeg-file", help="Path to EEG data file")
    parser.add_argument("--eeg-format", default="numpy", help="EEG file format (numpy, edf, gdf)")
    parser.add_argument("--channels", help="Comma-separated list of channel names")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency in Hz")
    
    parser.add_argument("--supabase-url", help="Supabase project URL")
    parser.add_argument("--supabase-key", help="Supabase API key")
    
    parser.add_argument("--gemini-api-key", help="Gemini API key")
    parser.add_argument("--gemini-model", default="gemini-pro", help="Gemini model name")
    
    parser.add_argument("--tokenizer", choices=["feature", "frequency", "autoencoder", "both", "all"], default="both",
                        help="Tokenizer type to use for training")
    parser.add_argument("--inference-tokenizer", choices=["feature", "frequency", "autoencoder"], default="frequency",
                        help="Tokenizer type to use for inference")
    parser.add_argument("--autoencoder-model-path", help="Path to pre-trained autoencoder model")
    
    parser.add_argument("--task", choices=["motor_imagery_classification", "thought_to_text", "abnormality_detection"], 
                        default="motor_imagery_classification", help="Analysis task to perform")
    
    parser.add_argument("--disable-rag", action="store_true", help="Disable RAG")
    
    parser.add_argument("--output", help="Path to output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference phase")
    
    args = parser.parse_args()
    
    # Check for autoencoder tokenizer without model path
    if (args.tokenizer == "autoencoder" or args.tokenizer == "all" or args.inference_tokenizer == "autoencoder") and not args.autoencoder_model_path:
        print("WARNING: Autoencoder tokenizer requested but no model path provided.")
        print("You can train an autoencoder model using examples/train_autoencoder.py")
    
    # Load or generate EEG data
    if args.eeg_file:
        print(f"Loading EEG data from {args.eeg_file}")
        eeg_data, fs, channel_names = load_eeg_data(args.eeg_file, args.eeg_format)
    else:
        # Use the default BCI dataset file instead of dummy data
        default_eeg_file = "BCI_IV_2a_EEGclip (2).npy"
        print(f"Loading default EEG data from {default_eeg_file}")
        eeg_data, fs, channel_names = load_eeg_data(default_eeg_file, "numpy")
        
    # Override sampling frequency if provided
    if args.fs:
        fs = args.fs
    
    # Override channel names if provided
    if args.channels:
        channel_names = args.channels.split(",")
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {channel_names}")
    
    # Set up Supabase client
    supabase_client = setup_supabase_client(args.supabase_url, args.supabase_key)
    
    # Set up Gemini client
    gemini_client = setup_gemini_client(args.gemini_api_key, args.gemini_model)
    
    # Initialize results
    results = {
        "training": None,
        "inference": None
    }
    
    # Run training phase
    if not args.skip_training and supabase_client:
        results["training"] = demo_training_phase(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=fs,
            supabase_client=supabase_client,
            tokenizer_type=args.tokenizer,
            autoencoder_model_path=args.autoencoder_model_path,
            debug=args.debug
        )
    
    # Run inference phase
    if not args.skip_inference and gemini_client:
        results["inference"] = demo_inference_phase(
            eeg_data=eeg_data,
            channel_names=channel_names,
            fs=fs,
            supabase_client=supabase_client,
            gemini_client=gemini_client,
            tokenizer_type=args.inference_tokenizer,
            autoencoder_model_path=args.autoencoder_model_path,
            task=args.task,
            use_rag=not args.disable_rag,
            debug=args.debug
        )
    
    # Save results if output file is specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                f.write(serializable_results)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 