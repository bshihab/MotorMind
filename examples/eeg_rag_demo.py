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
from vector_store.database.supabase_vector_store import SupabaseVectorStore
from training.data_processing.eeg_tokenizer_pipeline import EEGTokenizerPipeline
from inference.llm.eeg_rag_engine import EEGRAGEngine
from inference.llm.gemini_client import GeminiClient

# Import Supabase client
from vector_store.database.supabase_client import setup_supabase_client, SupabaseWrapper


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
    supabase = setup_supabase_client(url, key, debug=True)
    
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


def demo_training_phase(
    eeg_data, 
    channel_names, 
    fs, 
    supabase_client, 
    tokenizer_type="both",
    debug=False
):
    """
    Demonstrate the training phase: tokenization and vector store population.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        channel_names: List of channel names
        fs: Sampling frequency in Hz
        supabase_client: Initialized Supabase client
        tokenizer_type: Type of tokenizer to use ('feature', 'frequency', or 'both')
        debug: Whether to print debug information
        
    Returns:
        Training results
    """
    print("\n=== TRAINING PHASE ===")
    print(f"Tokenizing EEG data using {tokenizer_type} tokenizer...")
    
    # Initialize vector store
    vector_store = SupabaseVectorStore(
        supabase_client=supabase_client,
        table_name="eeg_embeddings",
        debug=debug
    )
    
    # Initialize tokenizer pipeline
    pipeline = EEGTokenizerPipeline(
        tokenizer_type=tokenizer_type,
        vector_store=vector_store,
        fs=fs,
        debug=debug
    )
    
    # Process the data
    start_time = time.time()
    results = pipeline.process_eeg_data(
        eeg_data=eeg_data,
        channel_names=channel_names,
        recording_id="demo_recording",
        metadata={"purpose": "demo", "recording_type": "motor_imagery"}
    )
    processing_time = time.time() - start_time
    
    # Print results
    print(f"Tokenization completed in {processing_time:.2f} seconds")
    
    if tokenizer_type == 'both':
        feature_count = results['tokenization'].get('feature', {}).get('count', 0)
        frequency_count = results['tokenization'].get('frequency', {}).get('count', 0)
        print(f"Generated {feature_count} feature tokens and {frequency_count} frequency tokens")
    else:
        token_count = results['tokenization'].get(tokenizer_type, {}).get('count', 0)
        print(f"Generated {token_count} {tokenizer_type} tokens")
    
    storage_success = results['storage'].get('success', False)
    storage_count = results['storage'].get('count', 0)
    
    if storage_success:
        print(f"Successfully stored {storage_count} token embeddings in the vector database")
    else:
        print("Failed to store token embeddings in the vector database")
    
    return results


def demo_inference_phase(
    eeg_data, 
    channel_names, 
    fs, 
    supabase_client, 
    gemini_client,
    tokenizer_type="frequency",
    task="motor_imagery_classification",
    use_rag=True,
    use_tree_of_thought=True,
    debug=False
):
    """
    Demonstrate the inference phase: tokenization, RAG, and LLM analysis.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        channel_names: List of channel names
        fs: Sampling frequency in Hz
        supabase_client: Initialized Supabase client
        gemini_client: Initialized Gemini client
        tokenizer_type: Type of tokenizer to use ('feature' or 'frequency')
        task: Analysis task (motor_imagery_classification, thought_to_text, etc.)
        use_rag: Whether to use RAG
        use_tree_of_thought: Whether to use tree-of-thought reasoning
        debug: Whether to print debug information
        
    Returns:
        Inference results
    """
    print("\n=== INFERENCE PHASE ===")
    print(f"Analyzing new EEG data using {tokenizer_type} tokenizer for {task} task...")
    
    if use_rag:
        print("RAG is enabled: Will use similar examples from the vector database")
    else:
        print("RAG is disabled: Will use LLM analysis only")
    
    if use_tree_of_thought:
        print("Tree-of-thought reasoning is enabled")
    
    # Initialize vector store
    vector_store = SupabaseVectorStore(
        supabase_client=supabase_client,
        table_name="eeg_embeddings",
        debug=debug
    )
    
    # Initialize inference engine
    engine = EEGRAGEngine(
        gemini_client=gemini_client,
        vector_store=vector_store if use_rag else None,
        tokenizer_type=tokenizer_type,
        fs=fs,
        debug=debug
    )
    
    # Process the data
    start_time = time.time()
    results = engine.process_eeg_data(
        eeg_data=eeg_data,
        channel_names=channel_names,
        task=task,
        use_rag=use_rag,
        use_tree_of_thought=use_tree_of_thought
    )
    processing_time = time.time() - start_time
    
    # Print results
    print(f"Inference completed in {processing_time:.2f} seconds")
    
    # Extract and print the interpretation
    interpretation = results.get('interpretation', {})
    if 'error' in interpretation:
        print(f"Error in interpretation: {interpretation['error']}")
    else:
        print("\nInterpretation Results:")
        if task == "thought_to_text":
            if 'text' in interpretation:
                print(f"Detected thought: {interpretation['text']}")
            if 'confidence' in interpretation and interpretation['confidence'] is not None:
                print(f"Confidence: {interpretation['confidence']*100:.1f}%")
        elif 'class' in interpretation:
            print(f"Classification: {interpretation['class']}")
            if 'confidence' in interpretation and interpretation['confidence'] is not None:
                print(f"Confidence: {interpretation['confidence']*100:.1f}%")
            if 'vote_count' in interpretation:
                print(f"Vote count: {interpretation['vote_count']}/{interpretation['total_votes']} "
                    f"({interpretation['vote_percentage']*100:.1f}%)")
    
    return results


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
    
    parser.add_argument("--tokenizer", choices=["feature", "frequency", "both"], default="both",
                        help="Tokenizer type to use for training")
    parser.add_argument("--inference-tokenizer", choices=["feature", "frequency"], default="frequency",
                        help="Tokenizer type to use for inference")
    
    parser.add_argument("--task", choices=["motor_imagery_classification", "thought_to_text", "abnormality_detection"], 
                        default="motor_imagery_classification", help="Analysis task to perform")
    
    parser.add_argument("--disable-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--disable-tot", action="store_true", help="Disable tree-of-thought reasoning")
    
    parser.add_argument("--output", help="Path to output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    parser.add_argument("--skip-training", action="store_true", help="Skip training phase")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference phase")
    
    args = parser.parse_args()
    
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
            task=args.task,
            use_rag=not args.disable_rag,
            use_tree_of_thought=not args.disable_tot,
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