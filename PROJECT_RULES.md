# MotorMind Project Rules and Standards

This document outlines the rules, standards, and architectural decisions for the MotorMind project.

## 1. Project Architecture

The project must maintain the following architecture:

```
MotorMind/
├── eeg_acquisition/      # EEG data loading and preprocessing
├── tokenization/         # Converting EEG signals to tokens
│   ├── feature_domain/   # Feature-based tokenization
│   ├── frequency_domain/ # Frequency-based tokenization
│   └── autoencoder/      # (Future) Autoencoder-based tokenization
├── vector_store/         # Storage for EEG token embeddings
│   └── database/         # Database integrations (Supabase)
├── training/             # Training pipelines and utilities
│   └── data_processing/  # Data processing pipelines
├── inference/            # Inference modules
│   └── llm/              # LLM integration for EEG analysis
├── examples/             # Example scripts
└── utils/                # Utility functions
```

## 2. Data Flow Rules

### Training Phase
The training phase must follow this sequence:
1. **Data Retrieval**: Load EEG data using the `eeg_acquisition` module
2. **Signal Processing**: Apply necessary preprocessing through the loaded modules
3. **Tokenization**: Convert to tokens using either feature-domain or frequency-domain tokenizers
4. **Vector Storage**: Store tokens in Supabase vector database
5. **RAG System**: Ensure all data is accessible for retrieval-augmented generation

### Inference Phase
The inference phase must follow this sequence:
1. **Data Retrieval**: Load new EEG data for analysis
2. **Signal Processing**: Apply the same preprocessing as training phase
3. **Tokenization**: Use the same tokenizer type as in training
4. **RAG Retrieval**: Retrieve similar examples from the vector store
5. **LLM Analysis**: Use Gemini (not ChatGPT/OpenAI) to interpret the data

## 3. LLM Integration Rules

1. **Use Google Gemini**: All LLM integrations must use Google Gemini, not OpenAI/ChatGPT
2. **API Key Storage**: The Gemini API key is stored in `gemini_api_key.txt`
3. **Key Loading**: All modules should load the key from this file, not hardcode it
4. **Error Handling**: Must gracefully handle API errors and rate limiting

## 4. Coding Standards

1. **Documentation**: All functions and classes must have docstrings
2. **Type Hints**: Use Python type hints for all function parameters and return values
3. **Error Handling**: Use try/except blocks for operations that might fail
4. **Logging**: Use the logging module, not print statements
5. **Tests**: Create unit tests for core functionality

## 5. Vector Store Rules

1. **Use Supabase**: All vector storage must use Supabase
2. **Embedding Format**: Embeddings must be normalized before storage
3. **Metadata**: Store appropriate metadata with each embedding
4. **Indexing**: Ensure proper indexing for efficient similarity search
5. **Error Recovery**: Implement retry mechanisms for database operations

## 6. File Format Standards

1. **EEG Data**: Use .npy format with shape (channels, samples)
2. **Default Dataset**: Use the BCI_IV_2a_EEGclip (2).npy file as default
3. **Configuration**: Use .env or specific config files, not hardcoded values
4. **Logging**: Use .log files for application logs

## 7. Performance Requirements

1. **Tokenization Speed**: Tokenization should process at least 10 seconds of EEG data per second
2. **Vector Search Speed**: Similarity searches should return in under 1 second
3. **Memory Usage**: Keep memory footprint below 4GB for normal operation

## 8. Security Rules

1. **API Keys**: Never commit API keys to version control
2. **Data Privacy**: Anonymize any patient/subject data
3. **Access Control**: Implement proper access controls for database operations

## 9. Contribution Guidelines

1. **Branch Naming**: feature/, bugfix/, hotfix/ prefixes
2. **Pull Requests**: Require review by at least one other contributor
3. **Issue Tracking**: Create issues for all planned changes
4. **Versioning**: Follow semantic versioning 