"""
LLM Wrapper Module

This module provides functionality to integrate Large Language Models (LLMs)
with EEG data analysis. It includes tools for:
1. Converting EEG features to text for LLM input
2. Generating prompts with few-shot examples
3. Parsing LLM responses
4. Using tree-of-thought reasoning for complex analysis
"""

import json
import os
import time
import re
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import requests
from pathlib import Path


class LLMWrapper:
    """
    Wrapper class for LLM integration with EEG data analysis.
    
    Supports multiple LLM backends and interfaces.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        few_shot_examples: Optional[List[Dict]] = None,
        examples_path: Optional[str] = None,
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            model_name: Name/identifier of the LLM to use
            api_key: API key for the LLM service
            api_base: Base URL for API requests
            max_tokens: Maximum tokens in LLM response
            temperature: Temperature parameter for response generation
            few_shot_examples: List of few-shot examples to include in prompts
            examples_path: Path to JSON file with few-shot examples
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load few-shot examples
        self.few_shot_examples = few_shot_examples or []
        if examples_path and not self.few_shot_examples:
            self._load_examples(examples_path)
    
    def _load_examples(self, path: str) -> None:
        """Load few-shot examples from a JSON file."""
        try:
            with open(path, 'r') as f:
                self.few_shot_examples = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading examples: {e}")
    
    def generate_prompt(
        self,
        feature_text: str,
        task: str = "motor_imagery_classification",
        n_examples: int = 3,
        instruction: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for the LLM with feature text and few-shot examples.
        
        Args:
            feature_text: Text representation of EEG features
            task: Task type for selecting relevant examples
            n_examples: Number of few-shot examples to include
            instruction: Optional custom instruction
            
        Returns:
            Complete prompt for the LLM
        """
        # Filter examples by task
        relevant_examples = [ex for ex in self.few_shot_examples if ex.get('task') == task]
        # Select a subset of examples
        selected_examples = relevant_examples[:min(n_examples, len(relevant_examples))]
        
        # Default instructions based on task
        if instruction is None:
            if task == "motor_imagery_classification":
                instruction = (
                    "Analyze the following EEG data features from motor imagery recording. "
                    "Classify the motor imagery as one of: right hand, left hand, feet, or rest. "
                    "Provide your reasoning in a step-by-step manner, and conclude with "
                    "a classification and confidence score."
                )
            elif task == "abnormality_detection":
                instruction = (
                    "Analyze the following EEG data features and determine if the recording shows "
                    "any abnormalities. Provide your reasoning step-by-step, listing any suspicious "
                    "patterns you identify. Conclude with a classification (normal/abnormal) and "
                    "confidence score."
                )
        
        # Construct prompt
        prompt_parts = [f"# Task: {instruction}\n"]
        
        # Add few-shot examples
        if selected_examples:
            prompt_parts.append("## Examples\n")
            for i, example in enumerate(selected_examples, 1):
                prompt_parts.append(f"### Example {i}\n")
                prompt_parts.append("Input:\n```")
                prompt_parts.append(example.get("input", ""))
                prompt_parts.append("```\n")
                prompt_parts.append("Output:\n```")
                prompt_parts.append(example.get("output", ""))
                prompt_parts.append("```\n")
        
        # Add the current input
        prompt_parts.append("## Current EEG Data Analysis\n")
        prompt_parts.append("```")
        prompt_parts.append(feature_text)
        prompt_parts.append("```\n")
        prompt_parts.append("Please analyze step-by-step:")
        
        return "\n".join(prompt_parts)
    
    def generate_tree_of_thought_prompt(
        self,
        feature_text: str,
        task: str = "motor_imagery_classification"
    ) -> str:
        """
        Generate a tree-of-thought reasoning prompt for complex EEG analysis.
        
        Args:
            feature_text: Text representation of EEG features
            task: Analysis task
            
        Returns:
            Tree-of-thought prompt
        """
        # Base instruction
        base_instruction = (
            "Analyze the following EEG data using tree-of-thought reasoning. "
            "For each step, consider multiple possible interpretations, evaluate each, "
            "and select the most likely one before proceeding to the next step."
        )
        
        # Task-specific instructions
        task_instructions = {
            "motor_imagery_classification": (
                "Your goal is to classify the motor imagery as right hand, left hand, feet, or rest.\n\n"
                "Step 1: Analyze the spatial pattern of alpha/beta power in sensorimotor cortex regions (C3, C4).\n"
                "Step 2: Check for event-related desynchronization or synchronization.\n"
                "Step 3: Compare contralateral vs ipsilateral activity patterns.\n"
                "Step 4: Consider alternative explanations for the observed patterns.\n"
                "Step 5: Make a classification with confidence score."
            ),
            "abnormality_detection": (
                "Your goal is to determine if the EEG shows any abnormal patterns.\n\n"
                "Step 1: Examine background rhythms for abnormalities.\n"
                "Step 2: Look for asymmetries between hemispheres.\n"
                "Step 3: Identify any paroxysmal activity.\n"
                "Step 4: Consider artifacts vs true abnormalities.\n"
                "Step 5: Classify as normal or abnormal with confidence score."
            )
        }
        
        # Construct prompt
        prompt_parts = [
            f"# Tree-of-Thought EEG Analysis\n",
            base_instruction + "\n",
            task_instructions.get(task, "Analyze the EEG data and provide your conclusions.") + "\n",
            "For each step, explicitly consider at least 2-3 possible interpretations before proceeding.",
            "\n## EEG Data\n```",
            feature_text,
            "```\n",
            "## Analysis"
        ]
        
        return "\n".join(prompt_parts)
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The input prompt for the LLM
            
        Returns:
            LLM response as a dictionary
        """
        # Validate API key
        if not self.api_key:
            return {
                "text": "Error: No API key provided. Set a valid API key using --llm-api-key flag or LLM_API_KEY environment variable.",
                "success": False,
                "error_type": "missing_api_key"
            }
            
        if self.api_key == "YOUR_ACTUAL_API_KEY" or self.api_key.lower().startswith("your_"):
            return {
                "text": "Error: Placeholder API key detected. Please replace 'YOUR_ACTUAL_API_KEY' with your real OpenAI API key.",
                "success": False,
                "error_type": "placeholder_api_key"
            }
            
        # Example implementation for OpenAI-compatible API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "text": result["choices"][0]["message"]["content"],
                "usage": result.get("usage", {}),
                "model": result.get("model", self.model_name),
                "success": True
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_msg = "Error: Invalid or expired API key. Please check your OpenAI API key."
                error_type = "invalid_api_key"
            elif e.response.status_code == 429:
                error_msg = "Error: Rate limit exceeded or insufficient quota. Check your OpenAI account."
                error_type = "rate_limit"
            else:
                error_msg = f"HTTP Error: {e}"
                error_type = "http_error"
            
            print(f"Error calling LLM API: {error_msg}")
            return {
                "text": error_msg,
                "success": False,
                "error_type": error_type
            }
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Failed to connect to the API. Check your internet connection."
            print(f"Error calling LLM API: {error_msg}")
            return {
                "text": error_msg,
                "success": False,
                "error_type": "connection_error"
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Error calling LLM API: {error_msg}")
            return {
                "text": error_msg,
                "success": False,
                "error_type": "unknown_error"
            }
    
    def analyze_eeg(
        self,
        feature_text: str,
        task: str = "motor_imagery_classification",
        use_tree_of_thought: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze EEG features using the LLM.
        
        Args:
            feature_text: Text representation of EEG features
            task: Analysis task
            use_tree_of_thought: Whether to use tree-of-thought reasoning
            
        Returns:
            Analysis results with classification and reasoning steps
        """
        # Generate prompt
        if use_tree_of_thought:
            prompt = self.generate_tree_of_thought_prompt(feature_text, task)
        else:
            prompt = self.generate_prompt(feature_text, task)
        
        # Call LLM
        response = self.call_llm(prompt)
        
        if not response["success"]:
            error_type = response.get("error_type", "unknown")
            error_message = response["text"]
            
            # Provide helpful instructions based on error type
            help_message = ""
            if error_type == "missing_api_key":
                help_message = "Please provide an API key using --llm-api-key or set the LLM_API_KEY environment variable."
            elif error_type == "placeholder_api_key":
                help_message = "Replace 'YOUR_ACTUAL_API_KEY' with your real OpenAI API key."
            elif error_type == "invalid_api_key":
                help_message = "Check that your API key is correct and has not expired."
            elif error_type == "rate_limit":
                help_message = "Wait and try again later or check your OpenAI account quota."
                
            if help_message:
                print(f"\nHelp: {help_message}")
                
            return {
                "classification": None,
                "confidence": None,
                "reasoning": None,
                "error": error_message,
                "error_type": error_type,
                "success": False
            }
        
        # Parse response
        result = self.parse_llm_response(response["text"], task)
        result["success"] = True
        
        return result
    
    def parse_llm_response(
        self,
        response_text: str,
        task: str = "motor_imagery_classification"
    ) -> Dict[str, Any]:
        """
        Parse the LLM response to extract classification, confidence, and reasoning.
        
        Args:
            response_text: Raw text response from the LLM
            task: Analysis task
            
        Returns:
            Parsed results with classification and reasoning steps
        """
        # Initialize result
        result = {
            "classification": None,
            "confidence": None,
            "reasoning": response_text,
            "success": True
        }
        
        # Extract classification and confidence
        if task == "motor_imagery_classification":
            # Check for the specific format in the Gemini output
            if "most likely classification is **right hand motor imagery**" in response_text:
                result["classification"] = "right hand"
            elif "most likely classification is **left hand motor imagery**" in response_text:
                result["classification"] = "left hand"
            elif "most likely classification is **feet motor imagery**" in response_text:
                result["classification"] = "feet"
            elif "most likely classification is **rest**" in response_text:
                result["classification"] = "rest"
            
            # If no exact match, try more general patterns
            if not result["classification"]:
                if "right hand" in response_text.lower():
                    result["classification"] = "right hand"
                elif "left hand" in response_text.lower():
                    result["classification"] = "left hand"
                elif "feet" in response_text.lower():
                    result["classification"] = "feet"
                elif "rest" in response_text.lower():
                    result["classification"] = "rest"
            
            # Look for confidence in various formats
            confidence_patterns = [
                r"\*\*confidence score:\*\*\s*(\d+)%",
                r"confidence score:?\s*low\s*\((\d+)%\)",
                r"confidence score:?\s*moderate\s*\((\d+)%\)",
                r"confidence score:?\s*high\s*\((\d+)%\)",
                r"confidence score:?\s*(\d+)%",
                r"confidence:?\s*(\d+)%",
                r"confidence:?\s*low\s*\((\d+)%\)",
                r"confidence:?\s*moderate\s*\((\d+)%\)",
                r"confidence:?\s*high\s*\((\d+)%\)",
                r"confidence:?\s*(\d+)\s*percent",
                r"confidence:?\s*(\d+)\.(\d+)",
                r"confidence score:?\s*(\d+)[\s\n]"
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, response_text.lower())
                if match:
                    try:
                        if len(match.groups()) == 1:
                            confidence = float(match.group(1))
                        else:
                            confidence = float(f"{match.group(1)}.{match.group(2)}")
                            
                        if confidence > 1 and confidence <= 100:
                            confidence /= 100
                        result["confidence"] = confidence
                        break
                    except (ValueError, IndexError):
                        pass
        
        elif task == "abnormality_detection":
            # Similar pattern matching for abnormality detection
            if "abnormal" in response_text.lower():
                result["classification"] = "abnormal"
            elif "normal" in response_text.lower():
                result["classification"] = "normal"
            
            # Look for confidence statements
            confidence_match = re.search(r"confidence:?\s*\w+\s*\((\d+)%\)", response_text.lower())
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    if confidence > 1 and confidence <= 100:
                        confidence /= 100
                    result["confidence"] = confidence
                except (ValueError, IndexError):
                    pass
        
        return result


class SupabaseLLMIntegration:
    """
    Class to integrate LLM-based EEG analysis with Supabase.
    
    Handles:
    1. Retrieving EEG data from Supabase
    2. Running LLM analysis
    3. Storing results back to Supabase
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        llm_wrapper: Optional[LLMWrapper] = None,
    ):
        """
        Initialize the Supabase-LLM integration.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            llm_wrapper: LLMWrapper instance
        """
        try:
            # Import here to avoid dependency issues
            from supabase import create_client
            self.supabase = create_client(supabase_url, supabase_key)
        except ImportError:
            print("Supabase Python client not installed. Run: pip install supabase")
            self.supabase = None
        
        self.llm_wrapper = llm_wrapper or LLMWrapper()
    
    def get_eeg_features(self, recording_id: str) -> List[Dict]:
        """
        Retrieve EEG features from Supabase for a specific recording.
        
        Args:
            recording_id: ID of the EEG recording
            
        Returns:
            List of feature dictionaries
        """
        if not self.supabase:
            return []
        
        try:
            # Query features from the database
            response = self.supabase.table("eeg_features") \
                .select("*") \
                .eq("recording_id", recording_id) \
                .order("window_start", ascending=True) \
                .execute()
            
            features = []
            for record in response.data:
                # Parse feature data from JSON if stored as string
                feature_data = record.get("feature_data")
                if isinstance(feature_data, str):
                    feature_data = json.loads(feature_data)
                
                features.append(feature_data)
            
            return features
        
        except Exception as e:
            print(f"Error retrieving EEG features: {e}")
            return []
    
    def analyze_and_store(
        self,
        recording_id: str,
        task: str = "motor_imagery_classification",
        use_tree_of_thought: bool = True,
        window_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze EEG data and store results in Supabase.
        
        Args:
            recording_id: ID of the EEG recording
            task: Analysis task
            use_tree_of_thought: Whether to use tree-of-thought reasoning
            window_index: Specific window to analyze (None for all)
            
        Returns:
            Analysis results
        """
        if not self.supabase:
            return {"error": "Supabase client not initialized", "success": False}
        
        # Get features
        features = self.get_eeg_features(recording_id)
        if not features:
            return {"error": "No features found", "success": False}
        
        # Select specific window or analyze all
        windows_to_analyze = [features[window_index]] if window_index is not None else features
        
        results = []
        for window_features in windows_to_analyze:
            # Convert features to text
            from ..data.preprocessing.features import features_to_text
            feature_text = features_to_text(window_features)
            
            # Analyze with LLM
            analysis = self.llm_wrapper.analyze_eeg(
                feature_text,
                task=task,
                use_tree_of_thought=use_tree_of_thought
            )
            
            # Add metadata
            analysis["recording_id"] = recording_id
            analysis["window_start"] = window_features.get("window_start")
            analysis["window_end"] = window_features.get("window_end")
            analysis["timestamp"] = time.time()
            
            # Store in Supabase
            try:
                self.supabase.table("model_predictions").insert({
                    "recording_id": recording_id,
                    "model_version": self.llm_wrapper.model_name,
                    "window_start": window_features.get("window_start"),
                    "window_end": window_features.get("window_end"),
                    "prediction": analysis.get("classification"),
                    "confidence": analysis.get("confidence"),
                    "explanation": analysis.get("reasoning"),
                    "created_at": 'now()'
                }).execute()
            except Exception as e:
                print(f"Error storing prediction: {e}")
                analysis["storage_error"] = str(e)
            
            results.append(analysis)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        } 