"""
LangChain Integration Module for MotorMind

This module provides a LangChain-based implementation for interfacing with LLMs
in the MotorMind project. It offers enhanced capabilities such as:
1. Advanced prompt management and chaining
2. Multiple LLM provider support
3. Structured reasoning chains
4. Memory for maintaining context
5. Tools for complex actions
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
from pathlib import Path

# LangChain imports
from langchain.llms import OpenAI, ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.callbacks import get_openai_callback

# Import base class for compatibility
from .llm_wrapper import LLMWrapper, SupabaseLLMIntegration


class LangChainWrapper(LLMWrapper):
    """
    LangChain implementation of the LLM wrapper for EEG data analysis.
    
    This extends the base LLMWrapper class to use LangChain's capabilities
    while maintaining API compatibility.
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
        use_memory: bool = False,
    ):
        """
        Initialize the LangChain wrapper.
        
        Args:
            model_name: Name/identifier of the LLM to use
            api_key: API key for the LLM service
            api_base: Base URL for API requests
            max_tokens: Maximum tokens in LLM response
            temperature: Temperature parameter for response generation
            few_shot_examples: List of few-shot examples to include in prompts
            examples_path: Path to JSON file with few-shot examples
            use_memory: Whether to use conversation memory for maintaining context
        """
        # Initialize the parent class
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=max_tokens,
            temperature=temperature,
            few_shot_examples=few_shot_examples,
            examples_path=examples_path,
        )
        
        # Initialize LangChain-specific attributes
        self.use_memory = use_memory
        self.memory = ConversationBufferMemory(memory_key="chat_history") if use_memory else None
        
        # Initialize the LLM based on the model type
        self._initialize_llm()
        
        # Create output parsers
        self._create_output_parsers()
    
    def _initialize_llm(self):
        """Initialize the appropriate LangChain LLM based on the model name."""
        # Determine if it's a chat model (like GPT-3.5/4) or a completion model
        is_chat_model = any(name in self.model_name.lower() for name in ["gpt-3.5", "gpt-4", "claude"])
        
        model_kwargs = {
            "model_name": self.model_name,
            "openai_api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add API base URL if provided
        if self.api_base:
            model_kwargs["openai_api_base"] = self.api_base
        
        # Initialize the appropriate model type
        if is_chat_model:
            self.llm = ChatOpenAI(**model_kwargs)
        else:
            self.llm = OpenAI(**model_kwargs)
    
    def _create_output_parsers(self):
        """Create structured output parsers for different analysis tasks."""
        # Define output schemas for different tasks
        motor_imagery_schemas = [
            ResponseSchema(name="classification", description="The motor imagery classification (right hand, left hand, feet, or rest)"),
            ResponseSchema(name="confidence", description="Confidence score from 0 to 1"),
            ResponseSchema(name="reasoning", description="Detailed reasoning steps that led to this classification")
        ]
        
        abnormality_schemas = [
            ResponseSchema(name="classification", description="Whether the EEG is normal or abnormal"),
            ResponseSchema(name="confidence", description="Confidence score from 0 to 1"),
            ResponseSchema(name="reasoning", description="Detailed reasoning steps that led to this conclusion")
        ]
        
        # Create parsers for each task
        self.output_parsers = {
            "motor_imagery_classification": StructuredOutputParser.from_response_schemas(motor_imagery_schemas),
            "abnormality_detection": StructuredOutputParser.from_response_schemas(abnormality_schemas)
        }
    
    def generate_prompt(
        self,
        feature_text: str,
        task: str = "motor_imagery_classification",
        n_examples: int = 3,
        instruction: Optional[str] = None
    ) -> ChatPromptTemplate:
        """
        Generate a LangChain prompt for the LLM with feature text and few-shot examples.
        
        Args:
            feature_text: Text representation of EEG features
            task: Task type for selecting relevant examples
            n_examples: Number of few-shot examples to include
            instruction: Optional custom instruction
            
        Returns:
            LangChain ChatPromptTemplate
        """
        # Use the parent class's logic to get the prompt text
        prompt_text = super().generate_prompt(
            feature_text=feature_text,
            task=task,
            n_examples=n_examples,
            instruction=instruction
        )
        
        # Convert to LangChain format
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "You are an expert neurologist analyzing EEG data. Provide detailed, step-by-step analysis."
        )
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            prompt_text + "\n\n" + self.output_parsers[task].get_format_instructions()
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        return chat_prompt
    
    def generate_tree_of_thought_prompt(
        self,
        feature_text: str,
        task: str = "motor_imagery_classification"
    ) -> ChatPromptTemplate:
        """
        Generate a tree-of-thought reasoning prompt using LangChain.
        
        Args:
            feature_text: Text representation of EEG features
            task: Analysis task
            
        Returns:
            LangChain ChatPromptTemplate
        """
        # Get the base tree-of-thought prompt text
        tot_prompt_text = super().generate_tree_of_thought_prompt(
            feature_text=feature_text,
            task=task
        )
        
        # Convert to LangChain format
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            "You are an expert neurologist analyzing EEG data. Use tree-of-thought reasoning to consider multiple interpretations at each step."
        )
        
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            tot_prompt_text + "\n\n" + self.output_parsers[task].get_format_instructions()
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
        
        return chat_prompt
    
    def call_llm(self, prompt: Union[str, ChatPromptTemplate]) -> Dict[str, Any]:
        """
        Call the LLM using LangChain.
        
        Args:
            prompt: Either a string or a LangChain prompt template
            
        Returns:
            LLM response as a dictionary
        """
        try:
            # If prompt is a string, convert it to a template
            if isinstance(prompt, str):
                prompt = PromptTemplate.from_template(prompt)
            
            # Create a chain
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory
            )
            
            # Track token usage
            with get_openai_callback() as cb:
                response = chain.run("")
                token_usage = {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_tokens": cb.total_tokens,
                    "cost": cb.total_cost
                }
            
            return {
                "text": response,
                "usage": token_usage,
                "model": self.model_name,
                "success": True
            }
            
        except Exception as e:
            print(f"Error calling LLM using LangChain: {e}")
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
        Analyze EEG features using LangChain.
        
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
        
        try:
            # Parse structured output
            parsed_output = self.output_parsers[task].parse(response["text"])
            # Convert confidence to float if needed
            if "confidence" in parsed_output and isinstance(parsed_output["confidence"], str):
                try:
                    confidence = float(parsed_output["confidence"])
                    # Normalize if needed
                    if confidence > 1 and confidence <= 100:
                        confidence /= 100
                    parsed_output["confidence"] = confidence
                except ValueError:
                    pass
            
            parsed_output["success"] = True
            return parsed_output
            
        except Exception as e:
            # If structured parsing fails, fall back to regex parsing
            print(f"Error parsing structured output: {e}. Falling back to regex parsing.")
            return self.parse_llm_response(response["text"], task)


class LangChainSupabaseIntegration(SupabaseLLMIntegration):
    """
    Integration of LangChain with Supabase for EEG analysis.
    
    This class extends the base Supabase integration to use LangChain
    for enhanced LLM capabilities.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        llm_wrapper: Optional[LangChainWrapper] = None,
    ):
        """
        Initialize the LangChain-Supabase integration.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            llm_wrapper: LangChainWrapper instance
        """
        # Create a LangChainWrapper if none provided
        if llm_wrapper is None:
            llm_wrapper = LangChainWrapper()
            
        # Initialize the parent class
        super().__init__(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            llm_wrapper=llm_wrapper
        )
    
    def analyze_with_agents(
        self,
        recording_id: str,
        task: str = "motor_imagery_classification",
        advanced_tools: bool = False,
        window_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Advanced analysis using LangChain agents with tools.
        
        Args:
            recording_id: ID of the EEG recording
            task: Analysis task
            advanced_tools: Whether to use advanced tools like web search
            window_index: Specific window to analyze (None for all)
            
        Returns:
            Analysis results
        """
        # This method would implement agent-based analysis
        # For now, we'll add a placeholder that calls the regular analyze_and_store
        # Can be expanded in the future
        
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
            
            # For now, just use the regular analyze_eeg method
            # This can be enhanced to use LangChain agents in the future
            analysis = self.llm_wrapper.analyze_eeg(
                feature_text,
                task=task,
                use_tree_of_thought=True
            )
            
            # Add metadata
            analysis["recording_id"] = recording_id
            analysis["window_start"] = window_features.get("window_start")
            analysis["window_end"] = window_features.get("window_end")
            analysis["timestamp"] = time.time()
            analysis["method"] = "langchain_agent" if advanced_tools else "langchain"
            
            # Store in Supabase
            try:
                self.supabase.table("model_predictions").insert({
                    "recording_id": recording_id,
                    "model_version": f"{self.llm_wrapper.model_name} (LangChain)",
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
            "count": len(results),
            "method": "langchain"
        }