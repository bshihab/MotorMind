"""
Gemini Wrapper Module

This module provides functionality to integrate Google's Gemini API
with EEG data analysis. It extends the base LLMWrapper with Gemini-specific
implementation details.
"""

import json
import os
from typing import Dict, List, Optional, Any

import requests
from .llm_wrapper import LLMWrapper
from ..utils.api_keys import get_gemini_api_key


class GeminiWrapper(LLMWrapper):
    """
    Wrapper class for Google's Gemini API integration with EEG data analysis.
    
    Extends the base LLMWrapper with Gemini-specific API calls.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        few_shot_examples: Optional[List[Dict]] = None,
        examples_path: Optional[str] = None,
    ):
        """
        Initialize the Gemini wrapper.
        
        Args:
            model_name: Name of the Gemini model to use (e.g., 'gemini-1.5-pro')
            api_key: Gemini API key
            max_tokens: Maximum tokens in response
            temperature: Temperature parameter for response generation
            few_shot_examples: List of few-shot examples to include in prompts
            examples_path: Path to JSON file with few-shot examples
        """
        # Try to get API key from file if not provided
        if api_key is None:
            api_key = get_gemini_api_key()
            
        # Initialize the parent LLMWrapper
        # Note: We don't need api_base for Gemini as we'll use the hardcoded endpoint
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            few_shot_examples=few_shot_examples,
            examples_path=examples_path,
        )
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call the Gemini API with the given prompt.
        
        Args:
            prompt: The input prompt for Gemini
            
        Returns:
            Gemini response as a dictionary
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
                "text": "Error: Placeholder API key detected. Please replace 'YOUR_ACTUAL_API_KEY' with your real Gemini API key.",
                "success": False,
                "error_type": "placeholder_api_key"
            }
        
        # Build the API endpoint URL with the API key
        api_endpoint = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent?key={self.api_key}"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare payload according to Gemini API format
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "topP": 0.9,
            }
        }
        
        try:
            # Send request to Gemini API
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract the text from Gemini's response format
            # The structure is different from OpenAI's
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    response_text = result['candidates'][0]['content']['parts'][0]['text']
                    return {
                        "text": response_text,
                        "usage": result.get("usageMetadata", {}),
                        "model": self.model_name,
                        "success": True
                    }
            
            # If we couldn't extract the text using the expected structure
            return {
                "text": "Error: Unexpected response format from Gemini API",
                "success": False,
                "error_type": "response_format_error"
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                error_msg = "Error: Invalid or expired API key. Please check your Gemini API key."
                error_type = "invalid_api_key"
            elif e.response.status_code == 429:
                error_msg = "Error: Rate limit exceeded. Check your Gemini API quota."
                error_type = "rate_limit"
            else:
                error_msg = f"HTTP Error: {e}"
                error_type = "http_error"
            
            print(f"Error calling Gemini API: {error_msg}")
            return {
                "text": error_msg,
                "success": False,
                "error_type": error_type
            }
        except requests.exceptions.ConnectionError:
            error_msg = "Error: Failed to connect to the Gemini API. Check your internet connection."
            print(f"Error calling Gemini API: {error_msg}")
            return {
                "text": error_msg,
                "success": False,
                "error_type": "connection_error"
            }
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Error calling Gemini API: {error_msg}")
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
        Analyze EEG features using the Gemini API.
        
        This overrides the parent method to add Gemini-specific error handling.
        
        Args:
            feature_text: Text representation of EEG features
            task: Analysis task
            use_tree_of_thought: Whether to use tree-of-thought reasoning
            
        Returns:
            Analysis results with classification and reasoning steps
        """
        # Generate prompt using the parent class methods
        if use_tree_of_thought:
            prompt = self.generate_tree_of_thought_prompt(feature_text, task)
        else:
            prompt = self.generate_prompt(feature_text, task)
        
        # Call Gemini API
        response = self.call_llm(prompt)
        
        if not response["success"]:
            error_type = response.get("error_type", "unknown")
            error_message = response["text"]
            
            # Provide helpful instructions based on error type
            help_message = ""
            if error_type == "missing_api_key":
                help_message = "Please provide an API key using --llm-api-key or set the LLM_API_KEY environment variable."
            elif error_type == "placeholder_api_key":
                help_message = "Replace 'YOUR_ACTUAL_API_KEY' with your real Gemini API key."
            elif error_type == "invalid_api_key":
                help_message = "Check that your Gemini API key is correct and has not expired."
            elif error_type == "rate_limit":
                help_message = "Wait and try again later or check your Gemini API quota."
                
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
        
        # Parse response using the parent class parser
        result = self.parse_llm_response(response["text"], task)
        result["success"] = True
        
        return result


class SupabaseGeminiIntegration:
    """
    Class to integrate Gemini-based EEG analysis with Supabase.
    
    This is a specialized version of SupabaseLLMIntegration that uses
    the GeminiWrapper instead of the generic LLMWrapper.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        gemini_wrapper: Optional[GeminiWrapper] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Supabase-Gemini integration.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            gemini_wrapper: GeminiWrapper instance
            api_key: Gemini API key (used if gemini_wrapper is not provided)
        """
        try:
            # Import here to avoid dependency issues
            from supabase import create_client
            self.supabase = create_client(supabase_url, supabase_key)
        except ImportError:
            print("Supabase Python client not installed. Run: pip install supabase")
            self.supabase = None
        
        # Use provided wrapper or create a new one
        self.gemini_wrapper = gemini_wrapper or GeminiWrapper(api_key=api_key)
    
    # Reuse methods from SupabaseLLMIntegration with Gemini wrapper
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
        Analyze EEG data with Gemini and store results in Supabase.
        
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
            # Import features_to_text function
            from ..data.preprocessing.features import features_to_text
            
            # Convert features to text
            feature_text = features_to_text(window_features)
            
            # Analyze with Gemini
            analysis = self.gemini_wrapper.analyze_eeg(
                feature_text,
                task=task,
                use_tree_of_thought=use_tree_of_thought
            )
            
            # Add metadata
            analysis["recording_id"] = recording_id
            analysis["window_start"] = window_features.get("window_start")
            analysis["window_end"] = window_features.get("window_end")
            
            # Store in Supabase
            try:
                self.supabase.table("model_predictions").insert({
                    "recording_id": recording_id,
                    "model_version": self.gemini_wrapper.model_name,
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