#!/usr/bin/env python
"""
MotorMind EEG-LLM LangChain Demo

This script demonstrates how to use the MotorMind system with LangChain
integration to analyze EEG data using Large Language Models.

Requirements:
- numpy
- scipy
- mne (for EEG data handling)
- supabase
- langchain
- openai
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
from ml.models.langchain_wrapper import LangChainWrapper, LangChainSupabaseIntegration
from examples.eeg_llm_demo import (
    load_eeg_data,
    create_dummy_eeg_data,
    process_eeg_data,
    setup_supabase_project,
    store_eeg_features
)
from backend.database.supabase import SupabaseClient


def analyze_with_langchain(
    feature_text,
    api_key=None,
    api_base="https://api.openai.com/v1",
    model_name="gpt-4",
    task="motor_imagery_classification",
    use_memory=False
):
    """
    Analyze EEG features using LangChain integration.
    
    Args:
        feature_text: Text representation of EEG features
        api_key: API key for LLM service
        api_base: API base URL
        model_name: LLM model name
        task: Analysis task
        use_memory: Whether to use conversation memory
        
    Returns:
        Analysis results
    """
    llm = LangChainWrapper(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1,
        use_memory=use_memory
    )
    
    result = llm.analyze_eeg(
        feature_text=feature_text,
        task=task,
        use_tree_of_thought=True
    )
    
    return result


def analyze_with_langchain_supabase(
    supabase_url,
    supabase_key,
    recording_id,
    api_key=None,
    api_base="https://api.openai.com/v1",
    model_name="gpt-4",
    task="motor_imagery_classification"
):
    """
    Analyze EEG data using LangChain and Supabase integration.
    
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
    llm = LangChainWrapper(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1
    )
    
    integration = LangChainSupabaseIntegration(
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


def analyze_with_langchain_agents(
    supabase_url,
    supabase_key,
    recording_id,
    api_key=None,
    api_base="https://api.openai.com/v1",
    model_name="gpt-4",
    task="motor_imagery_classification",
    advanced_tools=False
):
    """
    Analyze EEG data using LangChain agents with tools.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase API key
        recording_id: Recording ID
        api_key: API key for LLM service
        api_base: API base URL
        model_name: LLM model name
        task: Analysis task
        advanced_tools: Whether to use advanced tools
        
    Returns:
        Analysis results
    """
    llm = LangChainWrapper(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1
    )
    
    integration = LangChainSupabaseIntegration(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        llm_wrapper=llm
    )
    
    result = integration.analyze_with_agents(
        recording_id=recording_id,
        task=task,
        advanced_tools=advanced_tools
    )
    
    return result


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="MotorMind EEG-LLM LangChain Demo")
    parser.add_argument("--analysis-mode", choices=["direct", "supabase", "agent"], default="direct",
                        help="Analysis mode: direct, supabase, or agent")
    parser.add_argument("--supabase-url", help="Supabase project URL")
    parser.add_argument("--supabase-key", help="Supabase API key")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-model", default="gpt-4", help="LLM model name")
    parser.add_argument("--eeg-file", help="Path to EEG data file")
    parser.add_argument("--eeg-format", default="numpy", help="EEG file format (numpy, edf, gdf)")
    parser.add_argument("--output", default="langchain_results.json", help="Output file path")
    parser.add_argument("--use-memory", action="store_true", help="Use conversation memory")
    parser.add_argument("--advanced-tools", action="store_true", help="Use advanced tools with agents")
    
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
    
    if args.analysis_mode in ["supabase", "agent"]:
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
        
        if args.analysis_mode == "supabase":
            # Analyze with LangChain Supabase integration
            print("Analyzing with LangChain Supabase integration")
            results = analyze_with_langchain_supabase(
                supabase_url=args.supabase_url,
                supabase_key=args.supabase_key,
                recording_id=project_info["recording_id"],
                api_key=api_key,
                api_base=args.llm_api_base,
                model_name=args.llm_model,
                task="motor_imagery_classification"
            )
        else:  # agent mode
            # Analyze with LangChain agents
            print("Analyzing with LangChain agents")
            results = analyze_with_langchain_agents(
                supabase_url=args.supabase_url,
                supabase_key=args.supabase_key,
                recording_id=project_info["recording_id"],
                api_key=api_key,
                api_base=args.llm_api_base,
                model_name=args.llm_model,
                task="motor_imagery_classification",
                advanced_tools=args.advanced_tools
            )
        
    else:  # direct analysis
        # Direct analysis with LangChain
        print("Analyzing with LangChain")
        analysis = analyze_with_langchain(
            feature_text=feature_text,
            api_key=api_key,
            api_base=args.llm_api_base,
            model_name=args.llm_model,
            task="motor_imagery_classification",
            use_memory=args.use_memory
        )
        
        results = {
            "success": True,
            "results": [analysis],
            "count": 1,
            "method": "langchain_direct"
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