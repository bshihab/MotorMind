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
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return {
                "text": f"Error: {str(e)}",
                "success": False
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
            return {
                "classification": None,
                "confidence": None,
                "reasoning": None,
                "error": response["text"],
                "success": False
            }
        
        # Parse response
        return self.parse_llm_response(response["text"], task)
    
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
        # This is a simple heuristic-based approach - could be improved with regex or more sophisticated parsing
        if task == "motor_imagery_classification":
            classes = ["right hand", "left hand", "feet", "rest"]
            for line in response_text.lower().split("\n"):
                # Look for classification statements
                if "classification:" in line or "conclusion:" in line or "final classification:" in line:
                    for cls in classes:
                        if cls in line:
                            result["classification"] = cls
                            break
                
                # Look for confidence statements
                if "confidence:" in line:
                    try:
                        # Extract confidence value (e.g., "confidence: 85%" -> 0.85)
                        conf_text = line.split("confidence:")[1].strip()
                        conf_value = ''.join(filter(lambda x: x.isdigit() or x == '.', conf_text))
                        confidence = float(conf_value)
                        # Normalize to 0-1 range if needed
                        if confidence > 1 and confidence <= 100:
                            confidence /= 100
                        result["confidence"] = confidence
                    except (ValueError, IndexError):
                        pass
        
        elif task == "abnormality_detection":
            for line in response_text.lower().split("\n"):
                if "normal" in line or "abnormal" in line:
                    if "abnormal" in line:
                        result["classification"] = "abnormal"
                    else:
                        result["classification"] = "normal"
                
                # Look for confidence statements (same as above)
                if "confidence:" in line:
                    try:
                        conf_text = line.split("confidence:")[1].strip()
                        conf_value = ''.join(filter(lambda x: x.isdigit() or x == '.', conf_text))
                        confidence = float(conf_value)
                        if confidence > 1 and confidence <= 100:
                            confidence /= 100
                        result["confidence"] = confidence
                    except (ValueError, IndexError):
                        pass
        
        # Extract reasoning steps
        reasoning_lines = []
        in_reasoning = False
        
        for line in response_text.split("\n"):
            # Check for reasoning section markers
            if "reasoning:" in line.lower() or "analysis:" in line.lower() or "step " in line.lower():
                in_reasoning = True
                reasoning_lines.append(line)
            elif in_reasoning and line.strip():
                reasoning_lines.append(line)
            # End of reasoning section
            elif in_reasoning and "classification:" in line.lower() or "conclusion:" in line.lower():
                in_reasoning = False
        
        # If we found reasoning steps, update the reasoning field
        if reasoning_lines:
            result["reasoning"] = "\n".join(reasoning_lines)
        
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