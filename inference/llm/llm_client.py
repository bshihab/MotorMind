"""
Simple LLM Client

This module provides a simple client for interacting with Large Language Models
through various APIs (OpenAI, Azure, etc.).
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union
import requests


class SimpleLLMClient:
    """
    Simple client for interacting with Large Language Models.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4",
        api_base: str = "https://api.openai.com/v1",
        api_type: str = "openai",
        max_tokens: int = 2000,
        temperature: float = 0.2,
        timeout: int = 60,
        debug: bool = False
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            model_name: Name of the LLM model to use
            api_base: Base URL for API requests
            api_type: Type of API ('openai', 'azure', etc.)
            max_tokens: Maximum tokens in the response
            temperature: Temperature parameter for generation
            timeout: Request timeout in seconds
            debug: Whether to print debug information
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.api_base = api_base
        self.api_type = api_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.debug = debug
        
        if not self.api_key:
            if self.debug:
                print("Warning: No API key provided")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Generated response text
        """
        if self.api_type == "openai":
            return self._generate_openai(prompt)
        elif self.api_type == "azure":
            return self._generate_azure(prompt)
        else:
            if self.debug:
                print(f"Unsupported API type: {self.api_type}")
            return f"Error: Unsupported API type '{self.api_type}'"
    
    def _generate_openai(self, prompt: str) -> str:
        """
        Generate text using the OpenAI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.api_key:
            return "Error: No API key provided"
        
        try:
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
            
            endpoint = f"{self.api_base}/chat/completions"
            
            if self.debug:
                print(f"Sending request to {endpoint}")
                print(f"Model: {self.model_name}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if self.debug:
                    print(f"Response: {result}")
                
                # Extract the generated text
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    return "Error: No content in response"
            else:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                if self.debug:
                    print(error_msg)
                return f"Error: {error_msg}"
            
        except Exception as e:
            if self.debug:
                print(f"Exception in LLM request: {e}")
            return f"Error: {str(e)}"
    
    def _generate_azure(self, prompt: str) -> str:
        """
        Generate text using the Azure OpenAI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.api_key:
            return "Error: No API key provided"
        
        try:
            # Extract deployment name from model name
            deployment_name = self.model_name
            
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            endpoint = f"{self.api_base}/openai/deployments/{deployment_name}/chat/completions?api-version=2023-05-15"
            
            if self.debug:
                print(f"Sending request to {endpoint}")
                print(f"Deployment: {deployment_name}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if self.debug:
                    print(f"Response: {result}")
                
                # Extract the generated text
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    return "Error: No content in response"
            else:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                if self.debug:
                    print(error_msg)
                return f"Error: {error_msg}"
            
        except Exception as e:
            if self.debug:
                print(f"Exception in LLM request: {e}")
            return f"Error: {str(e)}"
    
    def stream_generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM with streaming.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Full generated response text
        """
        # This is a simplified implementation that doesn't actually stream
        # In a real implementation, you would yield chunks as they arrive
        return self.generate(prompt) 