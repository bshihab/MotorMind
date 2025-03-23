#!/usr/bin/env python

"""
MotorMind Autoencoder Training Script

This script demonstrates how to train an autoencoder tokenizer on EEG data,
which can then be used as a tokenization method in the MotorMind pipeline.
"""

import os
import sys
import numpy as np
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import MotorMind components
from eeg_acquisition.data_collection.eeg_loader import load_eeg_data, create_dummy_eeg_data
from tokenization.autoencoder.autoencoder_tokenizer import AutoencoderTokenizer


def prepare_training_data(eeg_data, fs, window_size=1.0, window_shift=0.5):
    """
    Prepare EEG data for autoencoder training by segmenting it.
    
    Args:
        eeg_data: EEG data with shape (channels, samples)
        fs: Sampling frequency in Hz
        window_size: Window size in seconds
        window_shift: Window shift in seconds
        
    Returns:
        List of EEG segments
    """
    # Calculate window and shift sizes in samples
    w_size = int(fs * window_size)
    w_shift = int(fs * window_shift)
    
    # Segment the data
    segments = []
    i = 0
    while i + w_size <= eeg_data.shape[1]:
        segments.append(eeg_data[:, i:i + w_size])
        i += w_shift
        
    print(f"Created {len(segments)} segments of shape {segments[0].shape}")
    return segments


def visualize_reconstruction(autoencoder, eeg_data, save_path=None):
    """
    Visualize original vs. reconstructed EEG signals.
    
    Args:
        autoencoder: Trained autoencoder
        eeg_data: EEG segment to reconstruct
        save_path: Path to save the visualization
    """
    # Ensure we have TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available for visualization")
        return
    
    # Preprocess for model
    model_input = np.expand_dims(eeg_data, axis=0)  # Add batch dimension
    
    # Get reconstruction
    reconstructed = autoencoder.model.predict(model_input)[0]
    
    # Plot some channels
    n_channels = min(4, eeg_data.shape[0])  # Show at most 4 channels
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10), sharex=True)
    
    for i in range(n_channels):
        ax = axes[i] if n_channels > 1 else axes
        
        ax.plot(eeg_data[i, :], 'b-', label='Original', alpha=0.7)
        ax.plot(reconstructed[i, :], 'r-', label='Reconstructed', alpha=0.7)
        ax.set_title(f"Channel {i}")
        ax.set_ylabel("Amplitude")
        ax.legend()
        
    plt.xlabel("Time (samples)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_latent_space(autoencoder, eeg_segments, save_path=None):
    """
    Visualize the latent space of the autoencoder.
    
    Args:
        autoencoder: Trained autoencoder
        eeg_segments: List of EEG segments
        save_path: Path to save the visualization
    """
    # Ensure we have TensorFlow
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not available for visualization")
        return
    
    # Generate embeddings for all segments
    embeddings = []
    for segment in eeg_segments[:100]:  # Limit to 100 segments for visualization
        try:
            # Add batch dimension
            model_input = np.expand_dims(segment, axis=0)
            # Get embedding from encoder
            embedding = autoencoder.encoder.predict(model_input, verbose=0)[0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding: {e}")
    
    if not embeddings:
        print("No valid embeddings generated for visualization")
        return
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Use t-SNE for dimensionality reduction if we have many dimensions
    if embeddings.shape[1] > 2:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
        except ImportError:
            print("scikit-learn not available for t-SNE visualization")
            return
    else:
        embeddings_2d = embeddings
    
    # Plot the 2D representation of the latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved latent space visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train an autoencoder tokenizer for EEG data")
    
    # Data parameters
    parser.add_argument("--eeg-file", help="Path to EEG data file")
    parser.add_argument("--eeg-format", default="numpy", help="EEG file format (numpy, edf, gdf)")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling frequency in Hz")
    
    # Autoencoder parameters
    parser.add_argument("--latent-dim", type=int, default=64, help="Dimension of the latent space")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    
    # Window parameters
    parser.add_argument("--window-size", type=float, default=1.0, help="Window size in seconds")
    parser.add_argument("--window-shift", type=float, default=0.5, help="Window shift in seconds")
    
    # Output parameters
    parser.add_argument("--model-output", default="models/autoencoder.keras", help="Path to save the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--vis-output-dir", default="visualizations", help="Directory to save visualizations")
    
    # Debug parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Load or generate EEG data
    if args.eeg_file:
        print(f"Loading EEG data from {args.eeg_file}")
        eeg_data, fs, channel_names = load_eeg_data(args.eeg_file, args.eeg_format)
    else:
        # Use the default BCI dataset file
        default_eeg_file = "BCI_IV_2a_EEGclip (2).npy"
        if os.path.exists(default_eeg_file):
            print(f"Loading default EEG data from {default_eeg_file}")
            eeg_data, fs, channel_names = load_eeg_data(default_eeg_file, "numpy")
        else:
            # Generate dummy data if no file is provided
            print("Generating dummy EEG data")
            n_channels = 22
            n_samples = int(args.fs * 60 * 5)  # 5 minutes of data
            eeg_data = create_dummy_eeg_data(n_channels, n_samples)
            fs = args.fs
            channel_names = [f"Ch{i}" for i in range(n_channels)]
    
    # Override sampling frequency if provided
    if args.fs:
        fs = args.fs
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {channel_names}")
    
    # Prepare data for autoencoder training
    eeg_segments = prepare_training_data(
        eeg_data, fs, 
        window_size=args.window_size,
        window_shift=args.window_shift
    )
    
    # Create autoencoder tokenizer
    autoencoder = AutoencoderTokenizer(
        input_shape=(eeg_data.shape[0], int(fs * args.window_size)),
        latent_dim=args.latent_dim,
        fs=fs,
        window_size=args.window_size,
        window_shift=args.window_shift,
        learning_rate=args.learning_rate,
        debug=args.debug
    )
    
    # Train the autoencoder
    print(f"\nTraining autoencoder with latent dimension {args.latent_dim}...")
    print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}")
    
    start_time = time.time()
    training_results = autoencoder.train(
        eeg_data=eeg_segments,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        save_path=args.model_output
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final training loss: {training_results.get('final_loss', 'N/A')}")
    
    # Create directory for visualizations if needed
    if args.visualize and not os.path.exists(args.vis_output_dir):
        os.makedirs(args.vis_output_dir)
    
    # Visualize results if requested
    if args.visualize:
        # Visualize reconstruction
        recon_path = os.path.join(args.vis_output_dir, "reconstruction.png")
        visualize_reconstruction(autoencoder, eeg_segments[0], recon_path)
        
        # Visualize latent space
        latent_path = os.path.join(args.vis_output_dir, "latent_space.png")
        visualize_latent_space(autoencoder, eeg_segments, latent_path)
    
    print("\nAutoencoder training completed!")
    print(f"Model saved to {args.model_output}")
    print("\nTo use this autoencoder in the MotorMind pipeline:")
    print(f"  python examples/eeg_rag_demo.py --tokenizer autoencoder --inference-tokenizer autoencoder --autoencoder-model-path {args.model_output}")


if __name__ == "__main__":
    main() 