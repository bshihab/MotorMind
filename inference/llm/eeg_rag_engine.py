"""
EEG-RAG Inference Engine

This module implements the inference engine that combines Google's Gemini with
Retrieval-Augmented Generation (RAG) to interpret EEG signals.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tokenization.feature_domain.feature_tokenizer import FeatureTokenizer
from tokenization.frequency_domain.frequency_tokenizer import FrequencyTokenizer
from vector_store.database.supabase_vector_store import SupabaseVectorStore


class EEGPromptGenerator:
    """
    Generates prompts for LLM analysis of EEG data, including context from RAG.
    """
    
    @staticmethod
    def generate_base_prompt(token_text: str, task: str = "motor_imagery_classification") -> str:
        """
        Generate a basic prompt for EEG analysis without RAG context.
        
        Args:
            token_text: Text representation of the EEG token
            task: Analysis task
            
        Returns:
            LLM prompt
        """
        if task == "motor_imagery_classification":
            instructions = (
                "Analyze the following EEG data features from motor imagery recording. "
                "Classify the motor imagery as one of: right hand, left hand, feet, or rest. "
                "Provide your reasoning in a step-by-step manner, and conclude with "
                "a classification and confidence score."
            )
        elif task == "thought_to_text":
            instructions = (
                "Analyze the following EEG frequency data to detect the word the subject might be thinking. "
                "Focus on language-related frequency patterns, particularly: "
                "1. Theta rhythms (4-8 Hz) which are critical for syllabic processing and verbal working memory, "
                "2. Alpha/Beta rhythms (8-30 Hz) involved in semantic processing, "
                "3. Gamma oscillations (30-80 Hz) associated with local word processing. "
                "\n\nPay special attention to channels over the left temporal and frontal areas (if present). "
                "Provide your reasoning in a step-by-step manner, and conclude with "
                "the most likely word or phrase the subject is thinking, with a confidence score."
            )
        elif task == "abnormality_detection":
            instructions = (
                "Analyze the following EEG data features and determine if the recording shows "
                "any abnormalities. Provide your reasoning step-by-step, listing any suspicious "
                "patterns you identify. Conclude with a classification (normal/abnormal) and "
                "confidence score."
            )
        else:
            instructions = f"Analyze the following EEG data for {task}. Provide your reasoning step-by-step."
        
        prompt = f"""# EEG Data Analysis Task

{instructions}

## EEG Data
```
{token_text}
```

## Analysis

Please analyze step-by-step:
1. 
"""
        
        return prompt
    
    @staticmethod
    def generate_rag_prompt(
        token_text: str,
        rag_context: List[Dict[str, Any]],
        task: str = "motor_imagery_classification",
        max_examples: int = 3
    ) -> str:
        """
        Generate a prompt for EEG analysis including RAG context.
        
        Args:
            token_text: Text representation of the EEG token
            rag_context: Similar tokens retrieved from RAG system
            task: Analysis task
            max_examples: Maximum number of RAG examples to include
            
        Returns:
            LLM prompt with RAG context
        """
        if task == "motor_imagery_classification":
            instructions = (
                "Analyze the following EEG data features from motor imagery recording. "
                "Classify the motor imagery as one of: right hand, left hand, feet, or rest. "
                "First look at the similar examples below, then analyze the new data. "
                "Provide your reasoning in a step-by-step manner, and conclude with "
                "a classification and confidence score."
            )
        elif task == "thought_to_text":
            instructions = (
                "Analyze the following EEG frequency data to detect the word the subject might be thinking. "
                "Focus on language-related frequency patterns, particularly: "
                "1. Theta rhythms (4-8 Hz) which are critical for syllabic processing and verbal working memory, "
                "2. Alpha/Beta rhythms (8-30 Hz) involved in semantic processing, "
                "3. Gamma oscillations (30-80 Hz) associated with local word processing. "
                "\n\nFirst examine the similar examples below, noting their associated words/phrases. "
                "Then analyze the new data, looking for frequency patterns that resemble known word patterns. "
                "Pay special attention to channels over the left temporal and frontal areas (if present). "
                "Provide your reasoning in a step-by-step manner, and conclude with "
                "the most likely word or phrase the subject is thinking, with a confidence score."
            )
        else:
            instructions = (
                f"Analyze the following EEG data for {task}. "
                f"First look at the similar examples below, then analyze the new data. "
                f"Provide your reasoning step-by-step."
            )
        
        # Build the RAG context part of the prompt
        rag_examples = []
        used_examples = rag_context[:max_examples]
        
        for i, example in enumerate(used_examples, 1):
            similarity = example.get("similarity", 0.0) * 100  # Convert to percentage
            
            # Extract metadata for label information
            metadata = example.get("metadata", {})
            label = metadata.get("label", "Unknown")
            
            rag_examples.append(f"### Example {i} (Similarity: {similarity:.1f}%)")
            rag_examples.append(f"Label: {label}")
            rag_examples.append("```")
            rag_examples.append(example.get("token_text", ""))
            rag_examples.append("```")
        
        rag_context_text = "\n".join(rag_examples)
        
        prompt = f"""# EEG Data Analysis with RAG

{instructions}

## Similar Examples from Database
{rag_context_text}

## New EEG Data for Analysis
```
{token_text}
```

## Analysis

Please analyze step-by-step:
1. 
"""
        
        return prompt
    
    @staticmethod
    def generate_tree_of_thought_prompt(
        token_text: str,
        rag_context: Optional[List[Dict[str, Any]]] = None,
        task: str = "motor_imagery_classification",
        max_examples: int = 2
    ) -> str:
        """
        Generate a tree-of-thought prompt for more complex EEG analysis.
        
        Args:
            token_text: Text representation of the EEG token
            rag_context: Similar tokens retrieved from RAG system (optional)
            task: Analysis task
            max_examples: Maximum number of RAG examples to include
            
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
            "thought_to_text": (
                "Your goal is to determine what word or phrase the subject is thinking based on EEG frequency data.\n\n"
                "Step 1: Analyze theta band (4-8 Hz) patterns related to syllabic processing and verbal working memory.\n"
                "Step 2: Examine alpha/beta bands (8-30 Hz) for semantic processing signatures.\n"
                "Step 3: Assess gamma oscillations (30-80 Hz) for local word processing.\n"
                "Step 4: Consider patterns across language-related brain regions (left temporal & frontal areas).\n"
                "Step 5: Identify potential words/phrases that match the observed patterns.\n"
                "Step 6: Determine the most likely word/phrase with a confidence score."
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
        
        specific_instructions = task_instructions.get(
            task, 
            f"Your goal is to analyze the EEG data for {task}.\n\n"
            "Step 1: Identify the key features in the data.\n"
            "Step 2: Consider different interpretations of these features.\n"
            "Step 3: Evaluate the evidence for each interpretation.\n"
            "Step 4: Draw a conclusion based on the most likely interpretation."
        )
        
        # Build the RAG context part of the prompt if provided
        rag_context_text = ""
        if rag_context:
            rag_examples = ["## Similar Examples from Database"]
            used_examples = rag_context[:max_examples]
            
            for i, example in enumerate(used_examples, 1):
                similarity = example.get("similarity", 0.0) * 100  # Convert to percentage
                
                # Extract metadata for label information
                metadata = example.get("metadata", {})
                label = metadata.get("label", "Unknown")
                
                rag_examples.append(f"### Example {i} (Similarity: {similarity:.1f}%)")
                rag_examples.append(f"Label: {label}")
                rag_examples.append("```")
                rag_examples.append(example.get("token_text", ""))
                rag_examples.append("```")
            
            rag_context_text = "\n".join(rag_examples) + "\n\n"
        
        # Construct the full prompt
        prompt = f"""# Tree-of-Thought EEG Analysis

{base_instruction}

{specific_instructions}

For each step, explicitly consider 2-3 possible interpretations before proceeding.

{rag_context_text}## EEG Data for Analysis
```
{token_text}
```

## Analysis

Step 1: Analyzing the patterns...
Interpretation A: 
"""
        
        return prompt


class EEGRAGEngine:
    """
    Inference engine that combines Gemini with RAG for EEG signal interpretation.
    """
    
    def __init__(
        self,
        gemini_client,
        vector_store: Optional[SupabaseVectorStore] = None,
        tokenizer_type: str = 'feature',
        tokenizer_params: Optional[Dict[str, Any]] = None,
        fs: float = 250,
        window_size: float = 1.0,
        window_shift: float = 0.1,
        debug: bool = False
    ):
        """
        Initialize the EEG-RAG inference engine.
        
        Args:
            gemini_client: Client for the Google Gemini API
            vector_store: Vector store instance for RAG
            tokenizer_type: Type of tokenizer to use ('feature' or 'frequency')
            tokenizer_params: Additional parameters for the tokenizer
            fs: Sampling frequency in Hz
            window_size: Window size in seconds
            window_shift: Window shift in seconds
            debug: Whether to print debug information
        """
        self.gemini_client = gemini_client
        self.vector_store = vector_store
        self.tokenizer_type = tokenizer_type
        self.fs = fs
        self.window_size = window_size
        self.window_shift = window_shift
        self.debug = debug
        
        # Default parameters for tokenizers
        self.tokenizer_params = tokenizer_params or {}
        
        # Initialize tokenizers
        self._init_tokenizers()
        
        # Initialize prompt generator
        self.prompt_generator = EEGPromptGenerator()
    
    def _init_tokenizers(self) -> None:
        """Initialize the tokenizers based on the selected type."""
        params = {
            'fs': self.fs,
            'window_size': self.window_size,
            'window_shift': self.window_shift,
            **self.tokenizer_params
        }
        
        if self.tokenizer_type == 'feature':
            self.tokenizer = FeatureTokenizer(**params)
        elif self.tokenizer_type == 'frequency':
            self.tokenizer = FrequencyTokenizer(**params)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
    
    def process_eeg_data(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        task: str = "motor_imagery_classification",
        use_rag: bool = True,
        use_tree_of_thought: bool = True,
        rag_top_k: int = 5,
        similarity_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Process EEG data through the inference pipeline.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            channel_names: List of channel names
            task: Analysis task
            use_rag: Whether to use RAG
            use_tree_of_thought: Whether to use tree-of-thought reasoning
            rag_top_k: Number of similar examples to retrieve from RAG
            similarity_threshold: Minimum similarity score for RAG
            
        Returns:
            Processing results
        """
        # Default channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
        
        # Initialize results
        results = {
            'tokenization': {},
            'embeddings': {},
            'rag': {'used': use_rag, 'similar_examples': []},
            'llm': {},
            'interpretation': {}
        }
        
        # Tokenize data
        tokens = self.tokenizer.tokenize(eeg_data, channel_names)
        results['tokenization'] = {
            'count': len(tokens),
            'method': self.tokenizer_type
        }
        
        # Process each token (typically multiple windows of the signal)
        token_results = []
        for token in tokens:
            # Generate embedding
            embedding = self.tokenizer.token_to_embedding(token)
            token_text = self.tokenizer.decode_token(token)
            
            # Query RAG if enabled
            rag_context = []
            if use_rag and self.vector_store is not None:
                rag_response = self.vector_store.query_similar(
                    query_embedding=embedding,
                    limit=rag_top_k,
                    similarity_threshold=similarity_threshold
                )
                
                if rag_response.get('success', False):
                    rag_context = rag_response.get('results', [])
            
            # Generate prompt based on settings
            if use_tree_of_thought:
                if use_rag and rag_context:
                    prompt = self.prompt_generator.generate_tree_of_thought_prompt(
                        token_text=token_text,
                        rag_context=rag_context,
                        task=task
                    )
                else:
                    prompt = self.prompt_generator.generate_tree_of_thought_prompt(
                        token_text=token_text,
                        task=task
                    )
            else:
                if use_rag and rag_context:
                    prompt = self.prompt_generator.generate_rag_prompt(
                        token_text=token_text,
                        rag_context=rag_context,
                        task=task
                    )
                else:
                    prompt = self.prompt_generator.generate_base_prompt(
                        token_text=token_text,
                        task=task
                    )
            
            # Call LLM for analysis
            llm_response = self.gemini_client.generate(prompt)
            
            # Extract interpretation from LLM response
            interpretation = self._parse_llm_response(llm_response, task)
            
            # Store results for this token
            token_result = {
                'window_start': token.get('window_start', 0),
                'window_end': token.get('window_end', 0),
                'token_text': token_text,
                'rag': {
                    'used': use_rag,
                    'similar_count': len(rag_context),
                    'similar_examples': rag_context
                },
                'llm': {
                    'prompt': prompt,
                    'response': llm_response
                },
                'interpretation': interpretation
            }
            
            token_results.append(token_result)
        
        # Aggregate results across all tokens
        results['token_results'] = token_results
        
        # Aggregate interpretations across all tokens
        if token_results:
            results['interpretation'] = self._aggregate_interpretations(token_results, task)
        
        return results
    
    def process_eeg_file(
        self,
        file_path: str,
        file_format: str = 'numpy',
        channel_names: Optional[List[str]] = None,
        task: str = "motor_imagery_classification",
        use_rag: bool = True,
        use_tree_of_thought: bool = True
    ) -> Dict[str, Any]:
        """
        Process EEG data from a file.
        
        Args:
            file_path: Path to the EEG data file
            file_format: Format of the file ('numpy', 'edf', 'gdf')
            channel_names: List of channel names
            task: Analysis task
            use_rag: Whether to use RAG
            use_tree_of_thought: Whether to use tree-of-thought reasoning
            
        Returns:
            Processing results
        """
        # Load the EEG data based on the file format
        eeg_data = self._load_eeg_data(file_path, file_format)
        
        if eeg_data is None:
            return {'error': f"Failed to load EEG data from {file_path}", 'success': False}
        
        # Process the data
        results = self.process_eeg_data(
            eeg_data=eeg_data,
            channel_names=channel_names,
            task=task,
            use_rag=use_rag,
            use_tree_of_thought=use_tree_of_thought
        )
        
        # Add file information
        results['file_info'] = {
            'path': file_path,
            'format': file_format,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else None,
            'modified': os.path.getmtime(file_path) if os.path.exists(file_path) else None
        }
        
        return results
    
    def _load_eeg_data(self, file_path: str, file_format: str) -> Optional[np.ndarray]:
        """
        Load EEG data from a file.
        
        Args:
            file_path: Path to the EEG data file
            file_format: Format of the file
            
        Returns:
            EEG data as numpy array or None if loading fails
        """
        try:
            if file_format == 'numpy':
                if file_path.endswith('.npz'):
                    with np.load(file_path) as data:
                        # Assuming the data is stored under a key like 'eeg_data'
                        # Modify this according to your actual data structure
                        for key in data.files:
                            if 'data' in key.lower() or 'eeg' in key.lower():
                                return data[key]
                        # If no suitable key is found, use the first one
                        return data[data.files[0]]
                else:  # .npy file
                    return np.load(file_path)
            
            elif file_format == 'edf':
                # Import mne here to avoid making it a required dependency
                try:
                    import mne
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    data = raw.get_data()
                    return data
                except ImportError:
                    if self.debug:
                        print("MNE package not found. Install with: pip install mne")
                    return None
            
            elif file_format == 'gdf':
                # Import mne here to avoid making it a required dependency
                try:
                    import mne
                    raw = mne.io.read_raw_gdf(file_path, preload=True)
                    data = raw.get_data()
                    return data
                except ImportError:
                    if self.debug:
                        print("MNE package not found. Install with: pip install mne")
                    return None
            
            else:
                if self.debug:
                    print(f"Unsupported file format: {file_format}")
                return None
        
        except Exception as e:
            if self.debug:
                print(f"Error loading EEG data: {e}")
            return None
    
    def _parse_llm_response(self, llm_response: str, task: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the interpretation.
        
        Args:
            llm_response: Response from the LLM
            task: Analysis task
            
        Returns:
            Parsed interpretation
        """
        # Initialize default interpretation
        interpretation = {
            'success': False,
            'class': None,
            'confidence': None,
            'reasoning': None,
            'raw_response': llm_response
        }
        
        try:
            # For motor imagery classification
            if task == "motor_imagery_classification":
                # Look for class and confidence statements
                class_patterns = [
                    r"classification:?\s*(right hand|left hand|feet|rest)",
                    r"classify\s*(?:as|:)?\s*(right hand|left hand|feet|rest)",
                    r"this is\s*(?:a)?\s*(right hand|left hand|feet|rest)"
                ]
                
                confidence_patterns = [
                    r"confidence:?\s*(\d+(?:\.\d+)?)%?",
                    r"confidence\s*(?:of|:|\s+)?\s*(\d+(?:\.\d+)?)%?",
                    r"(\d+(?:\.\d+)?)%?\s*confidence",
                    r"with\s*(?:a)?\s*confidence\s*(?:of)?\s*(\d+(?:\.\d+)?)%?"
                ]
                
                import re
                
                # Try to find the class
                for pattern in class_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        interpretation['class'] = match.group(1).lower()
                        break
                
                # Try to find the confidence
                for pattern in confidence_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        confidence = float(match.group(1))
                        # Normalize to 0-1 range if needed
                        if confidence > 1:
                            confidence /= 100
                        interpretation['confidence'] = confidence
                        break
                
                # Set success flag
                interpretation['success'] = interpretation['class'] is not None
                
                # Extract reasoning (everything before the conclusion)
                conclusion_markers = [
                    "conclusion:",
                    "in conclusion",
                    "therefore",
                    "classification:",
                    "final assessment:"
                ]
                
                reasoning_text = llm_response
                for marker in conclusion_markers:
                    if marker.lower() in llm_response.lower():
                        reasoning_text = llm_response.lower().split(marker.lower())[0]
                        break
                
                interpretation['reasoning'] = reasoning_text.strip()
            
            # For thought-to-text task
            elif task == "thought_to_text":
                import re
                
                # Look for word/phrase and confidence statements
                text_patterns = [
                    r"word:?\s*\"?([^\"\.]+)\"?",
                    r"phrase:?\s*\"?([^\"\.]+)\"?", 
                    r"thinking\s*(?:of|about)?:?\s*\"?([^\"\.]+)\"?",
                    r"detected\s*(?:word|phrase|thought):?\s*\"?([^\"\.]+)\"?",
                    r"most\s*likely\s*(?:word|phrase|thinking|thought):?\s*\"?([^\"\.]+)\"?",
                    r"subject\s*is\s*thinking\s*(?:of|about)?:?\s*\"?([^\"\.]+)\"?"
                ]
                
                # Same confidence patterns as before
                confidence_patterns = [
                    r"confidence:?\s*(\d+(?:\.\d+)?)%?",
                    r"confidence\s*(?:of|:|\s+)?\s*(\d+(?:\.\d+)?)%?",
                    r"(\d+(?:\.\d+)?)%?\s*confidence",
                    r"with\s*(?:a)?\s*confidence\s*(?:of)?\s*(\d+(?:\.\d+)?)%?"
                ]
                
                # Try to find the text
                for pattern in text_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        detected_text = match.group(1).strip()
                        # Remove extra quotes if present
                        detected_text = detected_text.strip('"\'')
                        interpretation['text'] = detected_text
                        break
                
                # Try to find the confidence
                for pattern in confidence_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        confidence = float(match.group(1))
                        # Normalize to 0-1 range if needed
                        if confidence > 1:
                            confidence /= 100
                        interpretation['confidence'] = confidence
                        break
                
                # Set success flag
                interpretation['success'] = 'text' in interpretation and interpretation['text'] is not None
                
                # Extract reasoning
                conclusion_markers = [
                    "conclusion:",
                    "in conclusion",
                    "therefore",
                    "final assessment:"
                ]
                
                reasoning_text = llm_response
                for marker in conclusion_markers:
                    if marker.lower() in llm_response.lower():
                        reasoning_text = llm_response.lower().split(marker.lower())[0]
                        break
                
                interpretation['reasoning'] = reasoning_text.strip()
            
            # For abnormality detection
            elif task == "abnormality_detection":
                # Look for abnormal/normal statements
                class_patterns = [
                    r"classification:?\s*(normal|abnormal)",
                    r"classify\s*(?:as|:)?\s*(normal|abnormal)",
                    r"the\s*eeg\s*is\s*(normal|abnormal)"
                ]
                
                # Implementation similar to motor imagery classification
                import re
                
                # Try to find the class
                for pattern in class_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        interpretation['class'] = match.group(1).lower()
                        break
                
                # Try to find the confidence (reuse existing patterns)
                confidence_patterns = [
                    r"confidence:?\s*(\d+(?:\.\d+)?)%?",
                    r"confidence\s*(?:of|:|\s+)?\s*(\d+(?:\.\d+)?)%?",
                    r"(\d+(?:\.\d+)?)%?\s*confidence",
                    r"with\s*(?:a)?\s*confidence\s*(?:of)?\s*(\d+(?:\.\d+)?)%?"
                ]
                
                for pattern in confidence_patterns:
                    match = re.search(pattern, llm_response, re.IGNORECASE)
                    if match:
                        confidence = float(match.group(1))
                        # Normalize to 0-1 range if needed
                        if confidence > 1:
                            confidence /= 100
                        interpretation['confidence'] = confidence
                        break
                
                # Set success flag
                interpretation['success'] = interpretation['class'] is not None
            
            # Generic parsing for other tasks
            else:
                # Just include the raw response
                interpretation['reasoning'] = llm_response
                interpretation['success'] = True
        
        except Exception as e:
            if self.debug:
                print(f"Error parsing LLM response: {e}")
            interpretation['error'] = str(e)
        
        return interpretation
    
    def _aggregate_interpretations(self, token_results: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        """
        Aggregate interpretations across multiple tokens.
        
        Args:
            token_results: Results for each token
            task: Analysis task
            
        Returns:
            Aggregated interpretation
        """
        # For motor imagery classification, use majority voting
        if task == "motor_imagery_classification":
            class_votes = {}
            confidence_sum = 0
            valid_interpretations = 0
            
            # Count votes for each class
            for result in token_results:
                interpretation = result.get('interpretation', {})
                if interpretation.get('success', False) and interpretation.get('class'):
                    class_name = interpretation['class']
                    class_votes[class_name] = class_votes.get(class_name, 0) + 1
                    
                    # Add confidence if available
                    if interpretation.get('confidence'):
                        confidence_sum += interpretation['confidence']
                        valid_interpretations += 1
            
            # Get the most voted class
            if class_votes:
                majority_class = max(class_votes.items(), key=lambda x: x[1])[0]
                majority_count = class_votes[majority_class]
                
                # Calculate average confidence
                avg_confidence = None
                if valid_interpretations > 0:
                    avg_confidence = confidence_sum / valid_interpretations
                
                return {
                    'class': majority_class,
                    'confidence': avg_confidence,
                    'vote_count': majority_count,
                    'total_votes': len(token_results),
                    'vote_percentage': majority_count / len(token_results),
                    'all_votes': class_votes
                }
            else:
                return {'error': "No valid interpretations found"}
        
        # For thought-to-text, use a different approach - find the most confident interpretation
        elif task == "thought_to_text":
            text_votes = {}
            confidence_by_text = {}
            
            # Count occurrences of each detected text and track confidences
            for result in token_results:
                interpretation = result.get('interpretation', {})
                if interpretation.get('success', False) and 'text' in interpretation:
                    detected_text = interpretation['text'].lower()  # Normalize to lowercase
                    text_votes[detected_text] = text_votes.get(detected_text, 0) + 1
                    
                    # Track confidence
                    if 'confidence' in interpretation and interpretation['confidence'] is not None:
                        if detected_text not in confidence_by_text:
                            confidence_by_text[detected_text] = []
                        confidence_by_text[detected_text].append(interpretation['confidence'])
            
            # If we have detected texts
            if text_votes:
                # Get the text with the most votes
                majority_text = max(text_votes.items(), key=lambda x: x[1])[0]
                majority_count = text_votes[majority_text]
                
                # Calculate average confidence for the majority text
                avg_confidence = None
                if majority_text in confidence_by_text:
                    confidences = confidence_by_text[majority_text]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                
                return {
                    'text': majority_text,
                    'confidence': avg_confidence,
                    'vote_count': majority_count,
                    'total_votes': len(token_results),
                    'vote_percentage': majority_count / len(token_results),
                    'all_detected_texts': text_votes
                }
            else:
                return {'error': "No valid thought-to-text interpretations found"}
        
        # For other tasks, return the most confident interpretation
        else:
            best_interpretation = None
            best_confidence = -1
            
            for result in token_results:
                interpretation = result.get('interpretation', {})
                if interpretation.get('success', False):
                    confidence = interpretation.get('confidence', 0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_interpretation = interpretation
            
            if best_interpretation:
                return best_interpretation
            else:
                return {'error': "No valid interpretations found"} 