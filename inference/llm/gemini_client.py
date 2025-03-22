"""
Gemini LLM Client

This module provides a client for interacting with Google's Gemini API.
It replaces the OpenAI/ChatGPT integration in the original MotorMind project.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
import requests
from pathlib import Path


class GeminiClient:
    """
    Client for interacting with Google's Gemini AI models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: int = 60,
        debug: bool = False
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: API key for Google Gemini API. If None, will try to load from gemini_api_key.txt
            model_name: Name of the model to use (gemini-pro, gemini-pro-vision, etc.)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            timeout: Request timeout in seconds
            debug: Whether to print debug information
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.debug = debug
        
        # Set API key
        self.api_key = api_key or self._load_api_key()
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided and couldn't be loaded from file")
        
        # Set base API URL
        self.api_base = "https://generativelanguage.googleapis.com/v1/models"
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from gemini_api_key.txt file.
        
        Returns:
            API key as string or None if not found
        """
        # Try to find the API key file in current directory or project root
        paths_to_try = [
            "gemini_api_key.txt",
            os.path.join(os.path.dirname(__file__), "gemini_api_key.txt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "gemini_api_key.txt")
        ]
        
        for path in paths_to_try:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        api_key = f.read().strip()
                        return api_key
            except Exception as e:
                if self.debug:
                    print(f"Error loading API key from {path}: {e}")
        
        # If we didn't find the key in files, try environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            return api_key
        
        if self.debug:
            print("Could not find Gemini API key in files or environment")
        
        return None
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Text prompt for generation
            
        Returns:
            Generated text response
        """
        try:
            url = f"{self.api_base}/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "topP": 0.95,
                    "topK": 64
                }
            }
            
            start_time = time.time()
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            if self.debug:
                print(f"Request took {time.time() - start_time:.2f}s")
            
            # Check for errors
            if response.status_code != 200:
                error_message = f"Gemini API error: {response.status_code} - {response.text}"
                if self.debug:
                    print(error_message)
                return f"Error: {error_message}"
            
            # Parse the response
            response_json = response.json()
            
            # Extract the generated text
            try:
                generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
                return generated_text
            except (KeyError, IndexError) as e:
                if self.debug:
                    print(f"Error parsing response: {e}")
                    print(f"Response: {response_json}")
                return f"Error parsing Gemini response: {e}"
            
        except Exception as e:
            if self.debug:
                print(f"Error calling Gemini API: {e}")
            return f"Error calling Gemini API: {e}"
    
    def stream_generate(self, prompt: str) -> str:
        """
        Generate text with streaming (simplified implementation for compatibility).
        
        Args:
            prompt: Text prompt for generation
            
        Returns:
            Generated text response
        """
        # Gemini API doesn't have a streaming interface that matches the
        # OpenAI client's interface, so we'll just use the regular generate
        # method for now. This ensures compatibility with existing code that
        # might expect a streaming interface.
        return self.generate(prompt)


# Alias for compatibility with the rest of the codebase
# This allows for drop-in replacement of the OpenAI client
SimpleLLMClient = GeminiClient 