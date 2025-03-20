# MotorMind

## Brain-Computer Interface with LLM Intelligence and Supabase Backend

MotorMind is a cutting-edge BCI (Brain-Computer Interface) project that leverages the power of Large Language Models (LLMs) to decode and interpret EEG motor data, with Supabase providing a robust data storage and management backend.

### Project Overview

Traditional approaches to EEG motor data analysis typically use CNNs or capsule networks, which require extensive labeled data and provide limited interpretability. MotorMind takes a novel approach by utilizing Large Language Models to:

1. Decode patterns in EEG motor data
2. Provide interpretable analysis through transparent reasoning
3. Adapt to individual users with minimal training data through few-shot learning

The project uses Supabase to:
- Store and manage EEG data securely
- Handle user profiles and authentication
- Track model performance and improvements over time
- Enable real-time collaborative research
- Provide API endpoints for application integration

### Technology Stack

- **EEG Data Processing**: Python, MNE, NumPy, SciPy
- **Machine Learning**: PyTorch, Transformers (for LLM integration)
- **Backend**: Supabase (PostgreSQL with extensions)
- **Frontend**: React with Supabase JS client
- **API**: FastAPI for model serving

### Research Foundation

This project builds upon research from "EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation" (arXiv:2401.18006v2), extending the approach to specifically focus on motor imagery data.

### Applications

- Assistive technology for individuals with motor disabilities
- Neurorehabilitation tools
- Human-computer interaction research
- Cognitive performance monitoring

### Setup and Installation

(Coming soon)

### Contribution

(Coming soon)

## LangChain Integration

The project now includes integration with LangChain, a powerful framework for developing applications powered by language models. This integration enhances MotorMind's capabilities in several ways:

### Features

- **Advanced Prompt Engineering**: Uses structured prompts with system and human messages for better LLM responses
- **Structured Output Parsing**: Automatically parses LLM responses into standardized formats
- **Conversation Memory**: Optionally maintains context across multiple interactions
- **Multiple LLM Support**: Works with various language model providers through LangChain's unified interface
- **Tree-of-Thought Reasoning**: Enhanced reasoning capabilities for complex EEG analysis

### Usage

The LangChain integration can be used in three modes:

1. **Direct Mode**: Analyzes EEG features directly using LangChain
2. **Supabase Mode**: Integrates with Supabase for data storage and retrieval
3. **Agent Mode**: Uses LangChain agents with tools for more advanced analysis (experimental)

Example usage:

```bash
# Direct analysis with LangChain
python examples/eeg_llm_langchain_demo.py --llm-api-key YOUR_API_KEY

# Analysis with Supabase integration
python examples/eeg_llm_langchain_demo.py --analysis-mode supabase --supabase-url YOUR_URL --supabase-key YOUR_KEY --llm-api-key YOUR_API_KEY

# Analysis with LangChain agents
python examples/eeg_llm_langchain_demo.py --analysis-mode agent --supabase-url YOUR_URL --supabase-key YOUR_KEY --llm-api-key YOUR_API_KEY --advanced-tools
```

### Implementation

The LangChain integration is implemented in:

- `ml/models/langchain_wrapper.py`: Contains the main implementation of the LangChain wrapper
- `examples/eeg_llm_langchain_demo.py`: Demonstrates how to use the LangChain integration

The implementation builds on the existing LLMWrapper architecture, extending it with LangChain's capabilities while maintaining compatibility with the rest of the codebase.

### Benefits

Using LangChain with MotorMind offers several advantages:

1. **Improved Response Quality**: Better structured prompts lead to more consistent and accurate analyses
2. **Enhanced Debugging**: Built-in token tracking and logging
3. **Future Extensibility**: Easy to add new features like tool use, retrieval augmentation, and more
4. **Framework Support**: Access to LangChain's growing ecosystem of extensions and tools

## Getting Started

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables or pass API keys as arguments.

3. Run one of the example scripts to see MotorMind in action.