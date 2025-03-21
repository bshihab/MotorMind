"""
Utility functions for handling API keys.
"""

import os
from pathlib import Path

def get_gemini_api_key():
    """
    Load the Gemini API key from the gemini_api_key.txt file.
    
    Returns:
        str: The Gemini API key or None if not found
    """
    # Possible locations for the API key file
    possible_paths = [
        "gemini_api_key.txt",                 # Current directory
        Path.home() / "gemini_api_key.txt",   # Home directory
        Path(__file__).parent.parent.parent / "gemini_api_key.txt"  # Project root
    ]
    
    # Try each path
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                api_key = f.read().strip()
                if api_key and not api_key.startswith("YOUR_"):
                    return api_key
        except FileNotFoundError:
            continue
    
    # Fall back to environment variable
    return os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") 