# MotorMind: EEG Signal Processing with RAG-LLM

MotorMind is a Python framework for processing EEG signals using a Retrieval-Augmented Generation (RAG) approach with Large Language Models (LLMs). It provides tools for EEG data acquisition, tokenization, vector storage, and inference through LLMs.

## Architecture

The project is organized into the following components:

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

### Key Components

1. **EEG Acquisition**: Tools for loading and preprocessing EEG data from various file formats
2. **Tokenization**: Converts raw EEG data into tokens using feature-based or frequency-based approaches
3. **Vector Store**: Stores EEG token embeddings in a vector database (Supabase)
4. **Training**: Processes EEG data and generates tokens for storage
5. **Inference**: Uses RAG with LLMs to interpret EEG signals

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MotorMind.git
cd MotorMind

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with the following variables:

```
# Supabase configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# OpenAI API configuration
OPENAI_API_KEY=your_openai_api_key
```

## Usage Example

Here's a basic example of using MotorMind to process EEG data and perform inference:

```bash
# Run with the default BCI dataset
python examples/eeg_rag_demo.py

# Run with specific EEG file
python examples/eeg_rag_demo.py --eeg_file path/to/your/eeg_data.npy --fs 250 --tokenizer frequency
```

### Using the Pipeline

The main pipeline consists of:

1. **Data Loading**: Load EEG data from the default BCI dataset or a specified file
2. **Tokenization**: Convert EEG signals to tokens using feature or frequency domain approaches
3. **Vector Storage**: Store tokens and their embeddings in Supabase
4. **Inference**: Query similar tokens and use LLM to interpret the EEG signals

## Tokenization Approaches

### Feature Domain Tokenizer

Extracts statistical and time-domain features from EEG signals:
- Signal statistics (mean, std, kurtosis, etc.)
- Peak analysis
- Hjorth parameters
- Sample entropy
- Zero crossings

### Frequency Domain Tokenizer

Analyzes frequency components of EEG signals:
- Fast Fourier Transform (FFT)
- Wavelet transform
- Band power extraction (delta, theta, alpha, beta, gamma)

## Development and Testing

Run tests with pytest:

```bash
pytest
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request