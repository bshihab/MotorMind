# LangChain Integration in MotorMind

This document provides detailed information about the LangChain integration in the MotorMind project, including its architecture, capabilities, and usage examples.

## Overview

LangChain is a framework designed to simplify the development of applications using large language models (LLMs). By integrating LangChain into MotorMind, we enhance the project's capabilities for EEG data analysis using LLMs, enabling more sophisticated prompt management, structured output parsing, and advanced reasoning techniques.

## Architecture

The LangChain integration is built on top of the existing `LLMWrapper` architecture, maintaining compatibility while adding new capabilities:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   EEG Features    │────▶│  LangChainWrapper │────▶│ Structured Output │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                               │        ▲
                               │        │
                               ▼        │
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│   Supabase DB     │◀───▶│ LangChainSupabase │     │ LangChain Models  │
│                   │     │   Integration     │────▶│  and Utilities    │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

### Key Components

1. **LangChainWrapper**: Extends the base `LLMWrapper` class to use LangChain's models, prompts, and output parsers.
2. **LangChainSupabaseIntegration**: Extends the `SupabaseLLMIntegration` class to integrate LangChain with Supabase.
3. **Structured Output Parsers**: Automatically extracts structured data from LLM responses.
4. **Advanced Prompt Templates**: Uses chat-based prompt templates with system and human messages.

## Features in Detail

### Advanced Prompt Engineering

LangChain enables more sophisticated prompt engineering through its specialized templates:

- **Chat Templates**: Creates structured chat prompts with system and human messages.
- **Few-shot Learning**: Efficiently manages few-shot examples in prompts.
- **Custom Instructions**: Easily customizable instructions for different analysis tasks.

Example:

```python
prompt = LangChainWrapper().generate_prompt(
    feature_text=eeg_feature_text,
    task="motor_imagery_classification",
    n_examples=3
)
```

### Structured Output Parsing

LangChain's output parsers help extract structured data from LLM responses:

- **Schema-based Parsing**: Defines expected output structure using schemas.
- **Automatic Type Conversion**: Handles data type conversion (e.g., string to float for confidence scores).
- **Fallback Mechanisms**: Falls back to regex parsing if structured parsing fails.

The integration defines schemas for different analysis tasks:

```python
motor_imagery_schemas = [
    ResponseSchema(name="classification", description="The motor imagery classification"),
    ResponseSchema(name="confidence", description="Confidence score from 0 to 1"),
    ResponseSchema(name="reasoning", description="Detailed reasoning steps")
]
```

### Memory and Context

LangChain provides capabilities for maintaining context across interactions:

- **Conversation Memory**: Stores previous interactions for context.
- **Stateful Analysis**: Allows building on previous analysis results.

Enable memory with:

```python
llm = LangChainWrapper(use_memory=True)
```

### Multiple LLM Support

LangChain's unified interface supports various LLM providers:

- **OpenAI Models**: GPT-3.5, GPT-4, etc.
- **Anthropic Models**: Claude, etc.
- **Open Source Models**: Support via compatible APIs.

The integration automatically selects the appropriate model type (chat vs. completion) based on the model name.

### Tree-of-Thought Reasoning

Enhanced reasoning for complex EEG analysis:

- **Multi-step Reasoning**: Breaks down analysis into explicit steps.
- **Alternative Interpretations**: Considers multiple possible interpretations at each step.

## Usage Examples

### Direct Analysis with LangChain

```python
from ml.models.langchain_wrapper import LangChainWrapper
from ml.data.preprocessing.features import features_to_text

# Initialize the wrapper
llm = LangChainWrapper(
    model_name="gpt-4",
    api_key="YOUR_API_KEY",
    temperature=0.1
)

# Convert features to text
feature_text = features_to_text(eeg_features)

# Analyze with LangChain
result = llm.analyze_eeg(
    feature_text=feature_text,
    task="motor_imagery_classification",
    use_tree_of_thought=True
)

# Access structured results
classification = result["classification"]
confidence = result["confidence"]
reasoning = result["reasoning"]
```

### Supabase Integration

```python
from ml.models.langchain_wrapper import LangChainWrapper, LangChainSupabaseIntegration

# Initialize LangChain wrapper
llm = LangChainWrapper(
    model_name="gpt-4",
    api_key="YOUR_API_KEY"
)

# Create Supabase integration
integration = LangChainSupabaseIntegration(
    supabase_url="YOUR_SUPABASE_URL",
    supabase_key="YOUR_SUPABASE_KEY",
    llm_wrapper=llm
)

# Analyze and store results
results = integration.analyze_and_store(
    recording_id="RECORDING_ID",
    task="motor_imagery_classification",
    use_tree_of_thought=True
)
```

### Command Line Usage

```bash
# Basic usage
python examples/eeg_llm_langchain_demo.py --llm-api-key YOUR_API_KEY

# With Supabase
python examples/eeg_llm_langchain_demo.py \
    --analysis-mode supabase \
    --supabase-url YOUR_SUPABASE_URL \
    --supabase-key YOUR_SUPABASE_KEY \
    --llm-api-key YOUR_API_KEY

# With conversation memory
python examples/eeg_llm_langchain_demo.py \
    --llm-api-key YOUR_API_KEY \
    --use-memory
```

## Advanced Features

### Agent-based Analysis (Experimental)

The integration includes experimental support for LangChain agents with tools:

```python
results = integration.analyze_with_agents(
    recording_id="RECORDING_ID",
    task="motor_imagery_classification",
    advanced_tools=True
)
```

This allows the LLM to:
- Research EEG patterns using web search tools
- Use calculators for more precise analysis
- Access reference databases for medical interpretations

## Implementation Details

### Key Files

- `ml/models/langchain_wrapper.py`: Main implementation
- `examples/eeg_llm_langchain_demo.py`: Usage examples

### Dependencies

The following dependencies are required:

```
langchain>=0.0.267
langchain-openai>=0.0.2
openai>=1.0.0
tiktoken>=0.5.1
```

## Future Enhancements

Planned enhancements for the LangChain integration include:

1. **Retrieval-Augmented Generation (RAG)**: Integrate with scientific literature on EEG patterns.
2. **Custom Tools**: Develop specialized tools for EEG analysis.
3. **Fine-tuned Models**: Support for fine-tuned models specific to EEG interpretation.
4. **Multi-modal Analysis**: Combine EEG data with other modalities using LangChain's multi-modal capabilities.

## Comparison with Base Implementation

| Feature                  | Base LLMWrapper        | LangChainWrapper        |
|--------------------------|------------------------|-------------------------|
| Prompt Management        | Basic text concatenation | Structured chat prompts |
| Output Parsing           | Regex-based            | Schema-based with fallback |
| Memory                   | Not supported          | Conversation memory     |
| Token Tracking           | Manual implementation  | Built-in support        |
| Provider Support         | OpenAI-like APIs       | Multiple providers      |
| Tool/Agent Support       | Not supported          | Experimental support    |
| Structured Reasoning     | Custom implementation  | Native support          |

## Conclusion

The LangChain integration enhances MotorMind's capabilities for EEG analysis with LLMs, providing a more structured and powerful framework for building advanced neural interpretation systems. It maintains compatibility with the existing codebase while adding valuable new features that improve the quality and capabilities of the analysis.