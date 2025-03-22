"""
Frequency-Domain EEG Tokenizer

This module implements a tokenization approach that converts EEG signals into
tokens based on frequency domain analysis (FFT/wavelet transforms).
"""

import numpy as np
from scipy import signal
import pywt
from typing import Dict, List, Any, Union, Optional, Tuple

class FrequencyTokenizer:
    """
    Tokenizes EEG data by analyzing frequency components using FFT and/or
    wavelet transforms, creating tokens based on significant frequency patterns.
    """
    
    def __init__(
        self,
        fs: float = 250,
        window_size: float = 1.0,
        window_shift: float = 0.1,
        frequency_bands: Optional[Dict[str, tuple]] = None,
        method: str = 'fft',  # 'fft', 'wavelet', or 'both'
        wavelet: str = 'db4'  # Wavelet type if using wavelet method
    ):
        """
        Initialize the frequency-domain tokenizer.
        
        Args:
            fs: Sampling frequency in Hz
            window_size: Window size in seconds
            window_shift: Window shift in seconds
            frequency_bands: Dictionary of frequency bands in the format {name: (low_freq, high_freq)}
            method: Frequency analysis method ('fft', 'wavelet', or 'both')
            wavelet: Wavelet type to use if method includes wavelet analysis
        """
        self.fs = fs
        self.window_size = window_size
        self.window_shift = window_shift
        
        # Define default frequency bands if not provided
        self.frequency_bands = frequency_bands or {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        self.method = method
        self.wavelet = wavelet
    
    def tokenize(self, eeg_data: np.ndarray, channel_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Tokenize EEG data using frequency domain analysis.
        
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
        
        # Extract frequency features for each segment
        tokens = []
        for i, segment in enumerate(segments):
            # Calculate window time in seconds
            window_start_time = i * self.window_shift
            window_end_time = window_start_time + self.window_size
            
            # Extract frequency features based on the selected method
            if self.method == 'fft':
                features = self._extract_fft_features(segment, channel_names)
            elif self.method == 'wavelet':
                features = self._extract_wavelet_features(segment, channel_names)
            else:  # 'both'
                fft_features = self._extract_fft_features(segment, channel_names)
                wavelet_features = self._extract_wavelet_features(segment, channel_names)
                # Merge the features
                features = {**fft_features, **wavelet_features}
            
            # Create token dictionary
            token = {
                'window_start': window_start_time,
                'window_end': window_end_time,
                'window_size': self.window_size,
                'features': features,
                'channel_names': channel_names,
                'method': self.method
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
        # Bandpass filter (0.5-45 Hz to keep all relevant frequency bands)
        filtered_data = self._bandpass_filter(eeg_data, 0.5, 45)
        
        # Z-score normalization
        normalized_data = self._normalize(filtered_data)
        
        return normalized_data
    
    def _bandpass_filter(
        self, 
        eeg_data: np.ndarray, 
        lowcut: float, 
        highcut: float, 
        order: int = 4
    ) -> np.ndarray:
        """Apply bandpass filter to EEG data."""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, eeg_data[ch, :])
            
        return filtered_data
    
    def _normalize(self, eeg_data: np.ndarray) -> np.ndarray:
        """Apply channel-wise z-score normalization."""
        return (eeg_data - eeg_data.mean(axis=1, keepdims=True)) / (eeg_data.std(axis=1, keepdims=True) + 1e-8)
    
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
    
    def _extract_fft_features(self, segment: np.ndarray, channel_names: List[str]) -> Dict[str, Any]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            Dictionary of frequency features
        """
        features = {
            'fft': {
                'band_power': {},
                'peak_frequency': {},
                'spectral_edge': {},
                'spectral_entropy': {}
            }
        }
        
        # Compute band power for each channel and frequency band
        for band_name, (low, high) in self.frequency_bands.items():
            features['fft']['band_power'][band_name] = {}
        
        # Process each channel
        for i, channel in enumerate(channel_names):
            # Get channel data
            channel_data = segment[i, :]
            
            # Apply window function to reduce spectral leakage
            windowed_data = channel_data * signal.windows.hann(len(channel_data))
            
            # Compute FFT
            fft_output = np.fft.rfft(windowed_data)
            fft_magnitude = np.abs(fft_output)
            
            # Get frequency bins
            freq_bins = np.fft.rfftfreq(len(channel_data), 1/self.fs)
            
            # Calculate spectral entropy
            pdf = fft_magnitude / np.sum(fft_magnitude + 1e-8)
            spectral_entropy = -np.sum(pdf * np.log2(pdf + 1e-8))
            features['fft']['spectral_entropy'][channel] = spectral_entropy
            
            # Calculate peak frequency
            peak_idx = np.argmax(fft_magnitude)
            peak_freq = freq_bins[peak_idx]
            features['fft']['peak_frequency'][channel] = peak_freq
            
            # Calculate spectral edge (frequency below which 95% of power is contained)
            cumulative_power = np.cumsum(fft_magnitude)
            total_power = cumulative_power[-1]
            if total_power > 0:
                spectral_edge_idx = np.where(cumulative_power >= 0.95 * total_power)[0][0]
                spectral_edge = freq_bins[spectral_edge_idx]
                features['fft']['spectral_edge'][channel] = spectral_edge
            
            # Calculate band power for each frequency band
            for band_name, (low, high) in self.frequency_bands.items():
                # Find frequencies in band
                mask = np.logical_and(freq_bins >= low, freq_bins <= high)
                
                if np.any(mask):
                    # Calculate average power in band
                    band_power = np.mean(fft_magnitude[mask]**2)
                    features['fft']['band_power'][band_name][channel] = band_power
        
        return features
    
    def _extract_wavelet_features(self, segment: np.ndarray, channel_names: List[str]) -> Dict[str, Any]:
        """
        Extract frequency domain features using wavelet transform.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            Dictionary of wavelet features
        """
        features = {
            'wavelet': {
                'coefficients': {},
                'energy': {},
                'entropy': {}
            }
        }
        
        # Process each channel
        for i, channel in enumerate(channel_names):
            # Get channel data
            channel_data = segment[i, :]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(channel_data, self.wavelet, level=5)
            
            # Store wavelet coefficients statistics for each level
            coeff_stats = {}
            energy = {}
            
            for j, coeff in enumerate(coeffs):
                if j == 0:
                    level_name = 'approximation'
                else:
                    level_name = f'detail_{j}'
                
                # Calculate statistics
                coeff_stats[level_name] = {
                    'mean': np.mean(coeff),
                    'std': np.std(coeff),
                    'kurtosis': float(np.mean((coeff - np.mean(coeff))**4) / (np.std(coeff)**4 + 1e-8)),
                    'max': np.max(np.abs(coeff))
                }
                
                # Calculate energy
                energy[level_name] = np.sum(coeff**2)
            
            # Store features
            features['wavelet']['coefficients'][channel] = coeff_stats
            features['wavelet']['energy'][channel] = energy
            
            # Calculate wavelet entropy
            total_energy = sum(energy.values())
            if total_energy > 0:
                relative_energy = {level: e / total_energy for level, e in energy.items()}
                entropy = -sum(re * np.log2(re + 1e-8) for re in relative_energy.values())
                features['wavelet']['entropy'][channel] = entropy
        
        return features
    
    def token_to_embedding(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Convert a token to a numerical embedding vector suitable for RAG.
        
        Args:
            token: Token dictionary with frequency features
            
        Returns:
            Embedding vector as a numpy array
        """
        # Extract feature values into a flat list
        features_flat = []
        features = token['features']
        
        # Add FFT features if present
        if 'fft' in features:
            # Add band power
            if 'band_power' in features['fft']:
                for band_name in self.frequency_bands:
                    if band_name in features['fft']['band_power']:
                        for channel, value in features['fft']['band_power'][band_name].items():
                            features_flat.append(value)
            
            # Add spectral entropy
            if 'spectral_entropy' in features['fft']:
                for channel, value in features['fft']['spectral_entropy'].items():
                    features_flat.append(value)
            
            # Add peak frequency
            if 'peak_frequency' in features['fft']:
                for channel, value in features['fft']['peak_frequency'].items():
                    features_flat.append(value / 100)  # Scale down frequencies
        
        # Add wavelet features if present
        if 'wavelet' in features:
            # Add wavelet entropy
            if 'entropy' in features['wavelet']:
                for channel, value in features['wavelet']['entropy'].items():
                    features_flat.append(value)
            
            # Add energy from different levels
            if 'energy' in features['wavelet']:
                for channel, level_energy in features['wavelet']['energy'].items():
                    total_energy = sum(level_energy.values()) + 1e-8
                    for level, energy in level_energy.items():
                        features_flat.append(energy / total_energy)  # Normalized energy
        
        # Convert to numpy array and normalize
        embedding = np.array(features_flat)
        if len(embedding) > 0:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        
        return embedding
    
    def decode_token(self, token: Dict[str, Any]) -> str:
        """
        Convert a token to a text representation suitable for LLM analysis.
        
        Args:
            token: Token dictionary with frequency features
            
        Returns:
            Text representation of the token
        """
        features = token['features']
        channel_names = token['channel_names']
        
        text_parts = ["EEG Frequency Analysis:"]
        
        # Add recording details
        text_parts.append(f"Window: {token.get('window_start', 0):.2f}s to {token.get('window_end', 2):.2f}s")
        text_parts.append(f"Analysis method: {token.get('method', 'unknown')}\n")
        
        # Add FFT features if present
        if 'fft' in features:
            text_parts.append("FFT Analysis:")
            
            # Add band power information
            if 'band_power' in features['fft']:
                text_parts.append("\nFrequency Band Power:")
                for band_name in self.frequency_bands:
                    if band_name in features['fft']['band_power']:
                        text_parts.append(f"\n{band_name.capitalize()} band power by channel:")
                        for channel, value in features['fft']['band_power'][band_name].items():
                            text_parts.append(f"  {channel}: {value:.2f}")
            
            # Add peak frequency
            if 'peak_frequency' in features['fft']:
                text_parts.append("\nPeak Frequency by channel:")
                for channel, value in features['fft']['peak_frequency'].items():
                    text_parts.append(f"  {channel}: {value:.2f} Hz")
            
            # Add spectral entropy
            if 'spectral_entropy' in features['fft']:
                text_parts.append("\nSpectral Entropy by channel:")
                for channel, value in features['fft']['spectral_entropy'].items():
                    text_parts.append(f"  {channel}: {value:.2f}")
        
        # Add wavelet features if present
        if 'wavelet' in features:
            text_parts.append("\nWavelet Analysis:")
            
            # Add wavelet entropy
            if 'entropy' in features['wavelet']:
                text_parts.append("\nWavelet Entropy by channel:")
                for channel, value in features['wavelet']['entropy'].items():
                    text_parts.append(f"  {channel}: {value:.2f}")
            
            # Add relative energy from different levels
            if 'energy' in features['wavelet']:
                text_parts.append("\nWavelet Energy Distribution:")
                for channel, level_energy in features['wavelet']['energy'].items():
                    text_parts.append(f"\n  Channel {channel}:")
                    total_energy = sum(level_energy.values()) + 1e-8
                    for level, energy in level_energy.items():
                        relative_energy = energy / total_energy * 100
                        text_parts.append(f"    {level}: {relative_energy:.1f}%")
        
        return "\n".join(text_parts) 