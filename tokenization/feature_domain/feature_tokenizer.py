"""
Feature-Based EEG Tokenizer

This module implements a tokenization approach that extracts meaningful features 
from EEG signals and converts them into tokens that can be used for RAG systems.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Any, Union, Optional
import json

class FeatureTokenizer:
    """
    Tokenizes EEG data by extracting meaningful features and creating embeddings.
    """
    
    def __init__(
        self,
        fs: float = 250,
        window_size: float = 1.0,
        window_shift: float = 0.1,
        frequency_bands: Optional[Dict[str, tuple]] = None
    ):
        """
        Initialize the feature-based tokenizer.
        
        Args:
            fs: Sampling frequency in Hz
            window_size: Window size in seconds
            window_shift: Window shift in seconds
            frequency_bands: Dictionary of frequency bands in the format {name: (low_freq, high_freq)}
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
    
    def tokenize(self, eeg_data: np.ndarray, channel_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Tokenize EEG data by extracting features and creating tokens.
        
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
        
        # Extract features for each segment
        tokens = []
        for i, segment in enumerate(segments):
            # Calculate window time in seconds
            window_start_time = i * self.window_shift
            window_end_time = window_start_time + self.window_size
            
            # Extract features
            features = self._extract_features(segment, channel_names)
            
            # Create token dictionary
            token = {
                'window_start': window_start_time,
                'window_end': window_end_time,
                'window_size': self.window_size,
                'features': features,
                'channel_names': channel_names
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
        # Bandpass filter (5-40 Hz)
        filtered_data = self._bandpass_filter(eeg_data, 5, 40)
        
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
    
    def _extract_features(self, segment: np.ndarray, channel_names: List[str]) -> Dict[str, Any]:
        """
        Extract comprehensive features from an EEG segment.
        
        Args:
            segment: EEG segment with shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract band power features
        band_powers = self._compute_band_power(segment, channel_names)
        features.update(band_powers)
        
        # Extract time domain features
        time_features = self._compute_time_domain_features(segment, channel_names)
        features.update(time_features)
        
        # Calculate alpha/beta ratio (important for motor imagery)
        alpha_beta_ratio = {}
        for channel in channel_names:
            if channel in band_powers['alpha_power'] and channel in band_powers['beta_power']:
                alpha = band_powers['alpha_power'][channel]
                beta = band_powers['beta_power'][channel]
                if beta > 0:
                    alpha_beta_ratio[channel] = alpha / beta
                else:
                    alpha_beta_ratio[channel] = 0
        features['alpha_beta_ratio'] = alpha_beta_ratio
        
        # Calculate motor imagery-specific features if C3 and C4 are present
        if 'C3' in channel_names and 'C4' in channel_names:
            c3_idx = channel_names.index('C3')
            c4_idx = channel_names.index('C4')
            
            # Calculate laterality
            if 'alpha_power' in band_powers:
                if 'C3' in band_powers['alpha_power'] and 'C4' in band_powers['alpha_power']:
                    c3_alpha = band_powers['alpha_power']['C3']
                    c4_alpha = band_powers['alpha_power']['C4']
                    sum_alpha = c3_alpha + c4_alpha
                    if sum_alpha > 0:
                        laterality = (c3_alpha - c4_alpha) / sum_alpha
                    else:
                        laterality = 0
                    features['alpha_laterality'] = laterality
        
        return features
    
    def _compute_band_power(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute power in different frequency bands for each channel.
        
        Args:
            eeg_data: EEG data of shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            Dictionary containing power values for each frequency band and channel
        """
        # Create result dictionary
        band_powers = {}
        for band_name in self.frequency_bands:
            band_powers[f"{band_name}_power"] = {}
        
        # Compute power for each channel
        for i, channel in enumerate(channel_names):
            # Get channel data
            channel_data = eeg_data[i, :]
            
            # Compute power spectral density
            f, psd = signal.welch(channel_data, self.fs, nperseg=min(256, len(channel_data)))
            
            # Calculate power in each band
            for band_name, (low, high) in self.frequency_bands.items():
                # Find frequencies in band
                idx_band = np.logical_and(f >= low, f <= high)
                
                # Calculate average power in band
                if np.any(idx_band):
                    power = np.mean(psd[idx_band])
                    band_powers[f"{band_name}_power"][channel] = power
        
        return band_powers
    
    def _compute_time_domain_features(self, eeg_data: np.ndarray, channel_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute time domain features for each channel.
        
        Args:
            eeg_data: EEG data of shape (channels, samples)
            channel_names: List of channel names
            
        Returns:
            Dictionary containing time domain features for each channel
        """
        # Create result dictionary
        features = {
            'mean': {},
            'std': {},
            'kurtosis': {},
            'skewness': {},
            'line_length': {},
            'zero_crossings': {}
        }
        
        # Compute features for each channel
        for i, channel in enumerate(channel_names):
            # Get channel data
            channel_data = eeg_data[i, :]
            
            # Calculate features
            features['mean'][channel] = np.mean(channel_data)
            features['std'][channel] = np.std(channel_data)
            features['kurtosis'][channel] = stats.kurtosis(channel_data)
            features['skewness'][channel] = stats.skew(channel_data)
            
            # Line length (sum of absolute differences)
            features['line_length'][channel] = np.sum(np.abs(np.diff(channel_data)))
            
            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
            features['zero_crossings'][channel] = len(zero_crossings)
        
        return features
    
    def token_to_embedding(self, token: Dict[str, Any]) -> np.ndarray:
        """
        Convert a token to a numerical embedding vector suitable for RAG.
        
        Args:
            token: Token dictionary with EEG features
            
        Returns:
            Embedding vector as a numpy array
        """
        # This is a simplified implementation - in practice, you might use
        # a more sophisticated approach to create embeddings
        
        # Extract feature values into a flat list
        features_flat = []
        
        # Add band power features
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            band_key = f"{band}_power"
            if band_key in token['features']:
                for channel, value in token['features'][band_key].items():
                    features_flat.append(value)
        
        # Add time domain features
        for feature in ['mean', 'std', 'kurtosis', 'skewness', 'line_length']:
            if feature in token['features']:
                for channel, value in token['features'][feature].items():
                    features_flat.append(value)
        
        # Convert to numpy array and normalize
        embedding = np.array(features_flat)
        if len(embedding) > 0:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        
        return embedding
    
    def decode_token(self, token: Dict[str, Any]) -> str:
        """
        Convert a token to a text representation suitable for LLM analysis.
        
        Args:
            token: Token dictionary with EEG features
            
        Returns:
            Text representation of the token
        """
        features = token['features']
        channel_names = token['channel_names']
        
        text_parts = ["EEG Features:"]
        
        # Add recording details
        text_parts.append(f"Window: {token.get('window_start', 0):.2f}s to {token.get('window_end', 2):.2f}s\n")
        
        # Add frequency band power features
        text_parts.append("Frequency Band Power:")
        
        # Add each frequency band
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            band_key = f"{band}_power"
            if band_key in features:
                text_parts.append(f"\n{band.capitalize()} band power by channel:")
                for channel, value in features[band_key].items():
                    text_parts.append(f"  {channel}: {value:.2f} µV²")
        
        # Add time domain features
        text_parts.append("\nTime Domain Features:")
        
        # Add line length (important for motor imagery)
        if 'line_length' in features:
            text_parts.append("\nLine length by channel:")
            for channel, value in features['line_length'].items():
                text_parts.append(f"  {channel}: {value:.2f}")
        
        # Add motor imagery specific features
        if 'alpha_laterality' in features:
            text_parts.append(f"\nAlpha laterality index: {features['alpha_laterality']:.3f}")
            text_parts.append("(Positive values indicate right hemisphere dominance, negative values indicate left hemisphere dominance)")
        
        if 'alpha_beta_ratio' in features:
            text_parts.append("\nAlpha/Beta ratio by channel:")
            for channel, value in features['alpha_beta_ratio'].items():
                text_parts.append(f"  {channel}: {value:.3f}")
        
        return "\n".join(text_parts) 