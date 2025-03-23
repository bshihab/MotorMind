"""
Autoencoder-Based EEG Tokenizer

This module implements a tokenization approach that uses deep learning autoencoders
to extract meaningful latent representations from EEG signals.
"""

import os
import numpy as np
import time
import pickle
import json
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing deep learning libraries with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Autoencoder will have limited functionality.")
    TF_AVAILABLE = False
    
class AutoencoderTokenizer:
    """
    Tokenizes EEG data by learning compressed representations using an autoencoder
    neural network architecture.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_shape: Tuple[int, int] = (22, 250),  # (channels, samples)
        latent_dim: int = 64,
        fs: float = 250.0,
        window_size: float = 1.0,
        window_shift: float = 0.1,
        learning_rate: float = 0.001,
        debug: bool = False
    ):
        """
        Initialize the autoencoder-based tokenizer.
        
        Args:
            model_path: Path to pre-trained model or None to create new
            input_shape: Input data shape (channels, samples)
            latent_dim: Size of the latent representation (embedding)
            fs: Sampling frequency in Hz
            window_size: Window size in seconds
            window_shift: Window shift in seconds
            learning_rate: Learning rate for model training
            debug: Whether to print debug information
        """
        self.fs = fs
        self.window_size = window_size
        self.window_shift = window_shift
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.debug = debug
        
        # Check TensorFlow availability
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Some functionality will be limited.")
            self.encoder = None
            self.decoder = None
            self.model = None
            return
            
        # Create or load model
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                # Extract encoder and decoder parts
                self._extract_encoder_decoder()
                logger.info(f"Loaded autoencoder model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self._create_model()
        else:
            self._create_model()
    
    def _create_model(self):
        """Create the autoencoder model architecture."""
        if not TF_AVAILABLE:
            logger.error("Cannot create model: TensorFlow not available")
            return
            
        try:
            # Flatten the input dimensions
            flattened_dim = self.input_shape[0] * self.input_shape[1]
            
            # Define encoder
            encoder_input = Input(shape=self.input_shape)
            x = Flatten()(encoder_input)
            x = Dense(1024, activation='relu')(x)
            x = Dense(256, activation='relu')(x)
            x = Dense(self.latent_dim, activation='linear', name='latent_layer')(x)
            self.encoder = Model(encoder_input, x, name='encoder')
            
            # Define decoder
            decoder_input = Input(shape=(self.latent_dim,))
            x = Dense(256, activation='relu')(decoder_input)
            x = Dense(1024, activation='relu')(x)
            x = Dense(flattened_dim, activation='linear')(x)
            x = Reshape(self.input_shape)(x)
            self.decoder = Model(decoder_input, x, name='decoder')
            
            # Define full autoencoder
            autoencoder_output = self.decoder(self.encoder(encoder_input))
            self.model = Model(encoder_input, autoencoder_output, name='autoencoder')
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse'
            )
            
            if self.debug:
                self.encoder.summary()
                self.decoder.summary()
                self.model.summary()
                
            logger.info("Created new autoencoder model")
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            
    def _extract_encoder_decoder(self):
        """Extract encoder and decoder models from the full autoencoder."""
        if not TF_AVAILABLE or self.model is None:
            return
            
        try:
            # Find the latent layer
            latent_layer = None
            for layer in self.model.layers:
                if isinstance(layer, Model) and layer.name == 'encoder':
                    self.encoder = layer
                elif isinstance(layer, Model) and layer.name == 'decoder':
                    self.decoder = layer
                    
            # If encoder/decoder not found as submodels, rebuild them
            if self.encoder is None or self.decoder is None:
                logger.warning("Could not extract encoder/decoder. Rebuilding models.")
                self._create_model()
        except Exception as e:
            logger.error(f"Error extracting encoder/decoder: {e}")
    
    def train(
        self, 
        eeg_data: Union[np.ndarray, List[np.ndarray]], 
        epochs: int = 50, 
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the autoencoder on EEG data.
        
        Args:
            eeg_data: EEG data with shape (samples, channels, time) or list of (channels, time)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data used for validation
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE or self.model is None:
            return {"error": "TensorFlow not available or model not initialized"}
        
        # Prepare the data
        try:
            if isinstance(eeg_data, list):
                # Handling list of segments
                X = np.array([self._preprocess_for_model(segment) for segment in eeg_data])
            else:
                # Handling ndarray with shape (samples, channels, time)
                X = eeg_data
            
            # Train the model
            history = self.model.fit(
                X, X,  # Autoencoder: input equals target
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1 if self.debug else 2
            )
            
            # Save model if path provided
            if save_path:
                self.model.save(save_path)
                logger.info(f"Model saved to {save_path}")
            
            return {
                "loss": history.history['loss'],
                "val_loss": history.history.get('val_loss', []),
                "epochs": epochs,
                "final_loss": history.history['loss'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"error": str(e)}
    
    def tokenize(self, eeg_data: np.ndarray, channel_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Tokenize EEG data using the autoencoder model.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            List of token dictionaries, each representing a window of EEG data
        """
        # Default channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
        
        # Preprocess the data
        preprocessed_data = self._preprocess(eeg_data)
        
        # Segment the data into windows
        segments = self._segment(preprocessed_data)
        
        # Extract embeddings for each segment
        tokens = []
        for i, segment in enumerate(segments):
            # Calculate window time in seconds
            window_start_time = i * self.window_shift
            window_end_time = window_start_time + self.window_size
            
            # Generate embedding
            embedding = self._generate_embedding(segment)
            
            # Create token dictionary
            token = {
                'window_start': window_start_time,
                'window_end': window_end_time,
                'window_size': self.window_size,
                'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                'channel_names': channel_names,
                'method': 'autoencoder'
            }
            
            tokens.append(token)
        
        return tokens
    
    def _preprocess(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data: bandpass filtering and normalization.
        
        Args:
            eeg_data: Raw EEG data
            
        Returns:
            Preprocessed EEG data
        """
        # Apply normalization (z-score)
        normalized_data = (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / (eeg_data.std(axis=1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _segment(self, eeg_data: np.ndarray) -> List[np.ndarray]:
        """Segment EEG data using sliding window."""
        w_size = int(self.fs * self.window_size)
        w_shift = int(self.fs * self.window_shift)
        segments = []
        
        i = 0
        while i + w_size <= eeg_data.shape[1]:
            segments.append(eeg_data[:, i:i + w_size])
            i += w_shift
            
        return segments
    
    def _preprocess_for_model(self, segment: np.ndarray) -> np.ndarray:
        """
        Prepare segment for input to the model.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            
        Returns:
            Reshaped segment suitable for model input
        """
        # Handle different segment shapes
        if segment.shape != self.input_shape:
            # Try to reshape or resample
            if segment.shape[0] == self.input_shape[0]:  # Same number of channels
                # Resample time dimension
                from scipy import signal
                resampled = []
                for ch in range(segment.shape[0]):
                    resampled.append(signal.resample(segment[ch, :], self.input_shape[1]))
                segment = np.array(resampled)
            else:
                raise ValueError(f"Segment shape {segment.shape} doesn't match expected {self.input_shape}")
                
        # Add batch dimension if needed for the model
        if TF_AVAILABLE:
            return np.expand_dims(segment, axis=0)
        return segment
    
    def _generate_embedding(self, segment: np.ndarray) -> np.ndarray:
        """
        Generate latent embedding from EEG segment.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            
        Returns:
            Latent embedding vector
        """
        if not TF_AVAILABLE or self.encoder is None:
            # Fallback to simple features if TensorFlow not available
            return self._generate_fallback_embedding(segment)
            
        try:
            # Prepare segment for model
            model_input = self._preprocess_for_model(segment)
            
            # Generate embedding
            embedding = self.encoder.predict(model_input, verbose=0)[0]  # Remove batch dimension
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._generate_fallback_embedding(segment)
    
    def _generate_fallback_embedding(self, segment: np.ndarray) -> np.ndarray:
        """
        Generate a fallback embedding when autoencoder is not available.
        This uses simple statistical features as a substitute.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            
        Returns:
            Simple feature vector
        """
        # Extract simple features
        features = []
        
        # Channel-wise features
        for ch in range(segment.shape[0]):
            # Time domain features
            features.append(np.mean(segment[ch, :]))
            features.append(np.std(segment[ch, :]))
            features.append(np.max(segment[ch, :]) - np.min(segment[ch, :]))
            
            # Simple frequency features using FFT
            fft_vals = np.abs(np.fft.rfft(segment[ch, :]))
            features.append(np.mean(fft_vals))
            features.append(np.sum(fft_vals[:5]))  # Low frequency power
            features.append(np.argmax(fft_vals))   # Peak frequency bin
        
        # Limit to latent_dim features
        embedding = np.array(features[:self.latent_dim])
        
        # Pad if needed
        if len(embedding) < self.latent_dim:
            embedding = np.pad(embedding, (0, self.latent_dim - len(embedding)))
            
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
        return embedding
    
    def token_to_embedding(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Convert a token to a numerical embedding vector suitable for RAG.
        
        Args:
            token: Token dictionary with embedding
            
        Returns:
            Embedding vector as a numpy array
        """
        # Extract embedding from token
        embedding = token.get('embedding', [])
        
        # Convert to numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        # Normalize if needed
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding
    
    def reconstruct(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct the original EEG signal from a token embedding.
        
        Args:
            token: Token dictionary with embedding
            
        Returns:
            Reconstructed EEG signal
        """
        if not TF_AVAILABLE or self.decoder is None:
            logger.error("Cannot reconstruct: TensorFlow not available or decoder not initialized")
            return np.array([])
            
        try:
            # Get embedding from token
            embedding = token.get('embedding', [])
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
                
            # Add batch dimension
            embedding = np.expand_dims(embedding, axis=0)
            
            # Reconstruct
            reconstructed = self.decoder.predict(embedding, verbose=0)[0]
            
            return reconstructed
        except Exception as e:
            logger.error(f"Error reconstructing signal: {e}")
            return np.array([])
    
    def decode_token(self, token: Dict[str, Any]) -> str:
        """
        Convert a token to a text representation suitable for LLM analysis.
        
        Args:
            token: Token dictionary with embedding
            
        Returns:
            Text representation of the token
        """
        embedding = token.get('embedding', [])
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        # Generate a description of the embedding
        text_parts = ["EEG Autoencoder Features:"]
        
        # Add recording details
        text_parts.append(f"Window: {token.get('window_start', 0):.2f}s to {token.get('window_end', 2):.2f}s")
        text_parts.append(f"Method: Autoencoder (latent dimension: {len(embedding)})\n")
        
        # Try to describe the embedding meaningfully
        try:
            # Add some statistics about the embedding
            text_parts.append("Latent Space Statistics:")
            text_parts.append(f"  Mean activation: {np.mean(embedding):.4f}")
            text_parts.append(f"  Activation std: {np.std(embedding):.4f}")
            text_parts.append(f"  Max activation: {np.max(embedding):.4f} at dimension {np.argmax(embedding)}")
            text_parts.append(f"  Min activation: {np.min(embedding):.4f} at dimension {np.argmin(embedding)}")
            
            # Add top activated dimensions
            top_dims = np.argsort(-np.abs(embedding))[:10]  # Top 10 by magnitude
            text_parts.append("\nTop Activated Dimensions:")
            for i, dim in enumerate(top_dims):
                text_parts.append(f"  Dimension {dim}: {embedding[dim]:.4f}")
                
            # If reconstruction is available, add some metrics
            if TF_AVAILABLE and self.decoder is not None:
                try:
                    # Reconstruct and calculate error
                    original_shape = (len(token.get('channel_names', [])), int(token.get('window_size', 1.0) * self.fs))
                    if hasattr(token, 'original_data'):
                        original = token['original_data']
                        reconstructed = self.reconstruct(token)
                        if len(original) > 0 and len(reconstructed) > 0:
                            mse = np.mean((original - reconstructed) ** 2)
                            text_parts.append(f"\nReconstruction MSE: {mse:.6f}")
                except:
                    pass
                    
        except Exception as e:
            # Fallback to simple representation
            text_parts.append("\nLatent Space Values (first 10):")
            for i in range(min(10, len(embedding))):
                text_parts.append(f"  Dimension {i}: {embedding[i]:.4f}")
        
        return "\n".join(text_parts)
    
    def save(self, path: str) -> bool:
        """
        Save the autoencoder model.
        
        Args:
            path: Path to save the model
            
        Returns:
            Success status
        """
        if not TF_AVAILABLE or self.model is None:
            logger.error("Cannot save: TensorFlow not available or model not initialized")
            return False
            
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def visualize_reconstruction(self, token: Dict[str, Any], path: str = None):
        """
        Visualize the original vs reconstructed EEG signal.
        
        Args:
            token: Token dictionary with embedding
            path: Path to save the visualization (if None, displays plot)
        """
        if not TF_AVAILABLE or self.decoder is None:
            logger.error("Cannot visualize: TensorFlow not available or decoder not initialized")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # Check if original data is available
            if not hasattr(token, 'original_data'):
                logger.error("Original data not available in token for comparison")
                return
                
            # Get original and reconstructed
            original = token['original_data']
            reconstructed = self.reconstruct(token)
            
            if len(original) == 0 or len(reconstructed) == 0:
                logger.error("Could not generate reconstruction")
                return
                
            # Plot comparison
            n_channels = min(4, original.shape[0])  # Show at most 4 channels
            fig, axes = plt.subplots(n_channels, 1, figsize=(10, 8), sharex=True)
            
            for i in range(n_channels):
                channel_name = token.get('channel_names', [f"Ch{i}"])[i]
                ax = axes[i] if n_channels > 1 else axes
                
                ax.plot(original[i, :], 'b-', label='Original', alpha=0.7)
                ax.plot(reconstructed[i, :], 'r-', label='Reconstructed', alpha=0.7)
                ax.set_title(f"Channel: {channel_name}")
                ax.set_ylabel("Amplitude")
                ax.legend()
                
            plt.xlabel("Time (samples)")
            plt.tight_layout()
            
            if path:
                plt.savefig(path)
                logger.info(f"Visualization saved to {path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing reconstruction: {e}") 