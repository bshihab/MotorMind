# MotorMind: EEG Signal Processing with RAG-Gemini

MotorMind is a Python framework for processing EEG signals using a Retrieval-Augmented Generation (RAG) approach with Google's Gemini AI. It provides tools for EEG data acquisition, tokenization, vector storage, and inference through Gemini.

## Architecture

The project is organized into the following components:

```
MotorMind/
├── eeg_acquisition/      # EEG data loading and preprocessing
├── tokenization/         # Converting EEG signals to tokens
│   ├── feature_domain/   # Feature-based tokenization
│   ├── frequency_domain/ # Frequency-based tokenization
│   └── autoencoder/      # Autoencoder-based tokenization
├── vector_store/         # Storage for EEG token embeddings
│   └── database/         # Database integrations (Supabase)
├── training/             # Training pipelines and utilities
│   └── data_processing/  # Data processing pipelines
├── inference/            # Inference modules
│   └── llm/              # Gemini integration for EEG analysis
├── examples/             # Example scripts
└── utils/                # Utility functions
```

### Key Components

1. **EEG Acquisition**: Tools for loading and preprocessing EEG data from various file formats
2. **Tokenization**: Converts raw EEG data into tokens using feature-based, frequency-based, or autoencoder-based approaches
3. **Vector Store**: Stores EEG token embeddings in a vector database (Supabase)
4. **Training**: Processes EEG data and generates tokens for storage
5. **Inference**: Uses RAG with Google Gemini to interpret EEG signals

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

# Gemini API configuration
GEMINI_API_KEY=your_gemini_api_key
```

Alternatively, you can store your credentials in text files in the project root:
- `supabase_url.txt` - Contains your Supabase URL
- `supabase_key.txt` - Contains your Supabase key
- `gemini_api_key.txt` - Contains your Gemini API key

## Usage Example

Here's a basic example of using MotorMind to process EEG data and perform inference:

```bash
# Run with the default BCI dataset
python examples/eeg_rag_demo.py

# Run with specific EEG file
python examples/eeg_rag_demo.py --eeg-file path/to/your/eeg_data.npy --fs 250 --tokenizer frequency

# Run with autoencoder tokenizer (after training)
python examples/eeg_rag_demo.py --tokenizer autoencoder --inference-tokenizer autoencoder --autoencoder-model-path models/autoencoder.keras
```

### Complete Pipeline with Autoencoder

To run the complete pipeline including autoencoder training:

```bash
# Train autoencoder, tokenize data, and run inference
python examples/run_motor_mind_pipeline.py

# With custom parameters
python examples/run_motor_mind_pipeline.py --latent-dim 32 --epochs 30 --task thought_to_text
```

### Training an Autoencoder

You can train an autoencoder separately:

```bash
# Train an autoencoder with visualizations
python examples/train_autoencoder.py --visualize

# With custom parameters
python examples/train_autoencoder.py --latent-dim 128 --epochs 50 --batch-size 64 --model-output models/custom_autoencoder.keras
```

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

### Autoencoder Tokenizer

Uses a neural network-based approach for tokenization:
- Compresses EEG signals into a lower-dimensional latent space
- Learns meaningful representations without manual feature engineering
- Provides better representation of complex non-linear patterns
- Capable of noise reduction and signal reconstruction

#### Advantages of Autoencoder Tokenization:
- Automatic feature extraction without domain expertise
- Potentially captures more complex patterns than traditional methods
- Compression of high-dimensional EEG data
- Ability to filter noise and reconstruct clean signals

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