"""
EEG Tokenization Pipeline

This module implements the training pipeline for processing EEG data,
tokenizing it, and preparing it for storage in the vector database.
"""

import os
import numpy as np
import json
from typing import Dict, List, Any, Union, Optional, Tuple
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from tokenization.feature_domain.feature_tokenizer import FeatureTokenizer
from tokenization.frequency_domain.frequency_tokenizer import FrequencyTokenizer
from tokenization.autoencoder.autoencoder_tokenizer import AutoencoderTokenizer
from vector_store.database.supabase_vector_store import SupabaseVectorStore


class EEGTokenizerPipeline:
    """
    Pipeline for processing EEG data, tokenizing it, and preparing it for
    storage in the vector database.
    """
    
    def __init__(
        self,
        tokenizer_type: str = 'feature',  # 'feature', 'frequency', 'autoencoder', or 'both'
        tokenizer_params: Optional[Dict[str, Any]] = None,
        vector_store: Optional[SupabaseVectorStore] = None,
        fs: float = 250,
        window_size: float = 1.0,
        window_shift: float = 0.1,
        debug: bool = False
    ):
        """
        Initialize the EEG tokenization pipeline.
        
        Args:
            tokenizer_type: Type of tokenizer to use
            tokenizer_params: Additional parameters for the tokenizer
            vector_store: Vector store instance for storing embeddings
            fs: Sampling frequency in Hz
            window_size: Window size in seconds
            window_shift: Window shift in seconds
            debug: Whether to print debug information
        """
        self.tokenizer_type = tokenizer_type
        self.tokenizer_params = tokenizer_params or {}
        self.vector_store = vector_store
        self.fs = fs
        self.window_size = window_size
        self.window_shift = window_shift
        self.debug = debug
        
        # Initialize default params with common settings
        default_params = {
            'fs': self.fs,
            'window_size': self.window_size,
            'window_shift': self.window_shift
        }
        
        # Merge default params with user-provided params
        self.tokenizer_params = {**default_params, **self.tokenizer_params}
        
        # Initialize tokenizers
        self.feature_tokenizer = None
        self.frequency_tokenizer = None
        self.autoencoder_tokenizer = None
        self._init_tokenizers()
        
    def _init_tokenizers(self) -> None:
        """Initialize the tokenizers based on the selected type."""
        if self.tokenizer_type in ['feature', 'both', 'all']:
            # Filter parameters for FeatureTokenizer
            feature_params = {
                'fs': self.tokenizer_params.get('fs', 250),
                'window_size': self.tokenizer_params.get('window_size', 1.0),
                'window_shift': self.tokenizer_params.get('window_shift', 0.1),
                'frequency_bands': self.tokenizer_params.get('frequency_bands', None)
            }
            self.feature_tokenizer = FeatureTokenizer(**feature_params)
            
        if self.tokenizer_type in ['frequency', 'both', 'all']:
            # Filter parameters for FrequencyTokenizer
            frequency_params = {
                'fs': self.tokenizer_params.get('fs', 250),
                'window_size': self.tokenizer_params.get('window_size', 1.0),
                'window_shift': self.tokenizer_params.get('window_shift', 0.1),
                'frequency_bands': self.tokenizer_params.get('frequency_bands', None)
            }
            self.frequency_tokenizer = FrequencyTokenizer(**frequency_params)
            
        if self.tokenizer_type in ['autoencoder', 'all']:
            # Get autoencoder-specific parameters
            ae_params = {
                'fs': self.tokenizer_params.get('fs', 250),
                'window_size': self.tokenizer_params.get('window_size', 1.0),
                'window_shift': self.tokenizer_params.get('window_shift', 0.1),
                'latent_dim': self.tokenizer_params.get('latent_dim', 64),
                'debug': self.debug
            }
            
            # Add model path if available
            if 'model_path' in self.tokenizer_params:
                ae_params['model_path'] = self.tokenizer_params['model_path']
                
            self.autoencoder_tokenizer = AutoencoderTokenizer(**ae_params)
            
    def process_eeg_data(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process EEG data through the tokenization pipeline.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            channel_names: List of channel names
            recording_id: Identifier for the recording
            user_id: Identifier for the user
            metadata: Additional metadata
            
        Returns:
            Dictionary with results
        """
        if self.debug:
            print(f"Processing EEG data with shape {eeg_data.shape}")
            print(f"Using tokenizer type: {self.tokenizer_type}")
        
        # Default channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
            
        # Default recording ID if not provided
        if recording_id is None:
            recording_id = f"recording_{int(time.time())}"
            
        # Initialize metadata
        if metadata is None:
            metadata = {}
            
        # Add basic metadata
        metadata.update({
            'recording_id': recording_id,
            'timestamp': time.time(),
            'fs': self.fs,
            'channels': len(channel_names),
            'duration': eeg_data.shape[1] / self.fs,
            'tokenizer_type': self.tokenizer_type
        })
        
        if user_id:
            metadata['user_id'] = user_id
            
        # Initialize results dictionary
        results = {
            'recording_id': recording_id,
            'tokenization': {},
            'storage': {'success': False, 'count': 0}
        }
        
        # Tokenize the data using selected tokenizer(s)
        tokens = []
        
        if self.tokenizer_type in ['feature', 'both', 'all'] and self.feature_tokenizer:
            start_time = time.time()
            feature_tokens = self.feature_tokenizer.tokenize(eeg_data, channel_names)
            tokenization_time = time.time() - start_time
            
            results['tokenization']['feature'] = {
                'count': len(feature_tokens),
                'time': tokenization_time
            }
            
            if self.debug:
                print(f"Generated {len(feature_tokens)} feature tokens in {tokenization_time:.2f} seconds")
                
            # Add tokens with tokenizer type
            for token in feature_tokens:
                token['tokenizer_type'] = 'feature'
                tokens.append(token)
                
        if self.tokenizer_type in ['frequency', 'both', 'all'] and self.frequency_tokenizer:
            start_time = time.time()
            frequency_tokens = self.frequency_tokenizer.tokenize(eeg_data, channel_names)
            tokenization_time = time.time() - start_time
            
            results['tokenization']['frequency'] = {
                'count': len(frequency_tokens),
                'time': tokenization_time
            }
            
            if self.debug:
                print(f"Generated {len(frequency_tokens)} frequency tokens in {tokenization_time:.2f} seconds")
                
            # Add tokens with tokenizer type
            for token in frequency_tokens:
                token['tokenizer_type'] = 'frequency'
                tokens.append(token)
                
        if self.tokenizer_type in ['autoencoder', 'all'] and self.autoencoder_tokenizer:
            start_time = time.time()
            autoencoder_tokens = self.autoencoder_tokenizer.tokenize(eeg_data, channel_names)
            tokenization_time = time.time() - start_time
            
            results['tokenization']['autoencoder'] = {
                'count': len(autoencoder_tokens),
                'time': tokenization_time
            }
            
            if self.debug:
                print(f"Generated {len(autoencoder_tokens)} autoencoder tokens in {tokenization_time:.2f} seconds")
                
            # Add tokens with tokenizer type
            for token in autoencoder_tokens:
                token['tokenizer_type'] = 'autoencoder'
                tokens.append(token)
        
        # Store tokenized data if vector store is provided
        if self.vector_store and tokens:
            storage_results = self._store_tokens(tokens, recording_id, metadata)
            results['storage'] = storage_results
            
        return results
    
    def process_eeg_file(
        self,
        file_path: str,
        file_format: str = 'numpy',
        channel_names: Optional[List[str]] = None,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process EEG data from a file.
        
        Args:
            file_path: Path to the EEG data file
            file_format: Format of the file ('numpy', 'edf', 'gdf')
            channel_names: List of channel names
            recording_id: ID of the recording
            user_id: ID of the user
            metadata: Additional metadata
            
        Returns:
            Processing results
        """
        # Load the EEG data based on the file format
        eeg_data = self._load_eeg_data(file_path, file_format)
        
        if eeg_data is None:
            return {'error': f"Failed to load EEG data from {file_path}", 'success': False}
        
        # Get metadata from the file if not provided
        if metadata is None:
            metadata = self._extract_file_metadata(file_path)
        
        # Process the data
        results = self.process_eeg_data(
            eeg_data=eeg_data,
            channel_names=channel_names,
            recording_id=recording_id or os.path.basename(file_path),
            user_id=user_id,
            metadata=metadata
        )
        
        return results
    
    def _load_eeg_data(self, file_path: str, file_format: str) -> Optional[np.ndarray]:
        """
        Load EEG data from a file.
        
        Args:
            file_path: Path to the EEG data file
            file_format: Format of the file
            
        Returns:
            EEG data as numpy array or None if loading fails
        """
        try:
            if file_format == 'numpy':
                if file_path.endswith('.npz'):
                    with np.load(file_path) as data:
                        # Assuming the data is stored under a key like 'eeg_data'
                        # Modify this according to your actual data structure
                        for key in data.files:
                            if 'data' in key.lower() or 'eeg' in key.lower():
                                return data[key]
                        # If no suitable key is found, use the first one
                        return data[data.files[0]]
                else:  # .npy file
                    return np.load(file_path)
            
            elif file_format == 'edf':
                # Import mne here to avoid making it a required dependency
                try:
                    import mne
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                    data = raw.get_data()
                    return data
                except ImportError:
                    if self.debug:
                        print("MNE package not found. Install with: pip install mne")
                    return None
            
            elif file_format == 'gdf':
                # Import mne here to avoid making it a required dependency
                try:
                    import mne
                    raw = mne.io.read_raw_gdf(file_path, preload=True)
                    data = raw.get_data()
                    return data
                except ImportError:
                    if self.debug:
                        print("MNE package not found. Install with: pip install mne")
                    return None
            
            else:
                if self.debug:
                    print(f"Unsupported file format: {file_format}")
                return None
        
        except Exception as e:
            if self.debug:
                print(f"Error loading EEG data: {e}")
            return None
    
    def _extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the file name and properties.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Metadata dictionary
        """
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1]
        file_modified = os.path.getmtime(file_path)
        
        metadata = {
            'file_name': file_name,
            'file_size': file_size,
            'file_extension': file_extension,
            'file_modified': file_modified,
            'processing_time': time.time()
        }
        
        # Try to extract additional metadata if it's a numpy .npz file
        if file_path.endswith('.npz'):
            try:
                with np.load(file_path) as data:
                    # Look for metadata keys
                    for key in data.files:
                        if 'meta' in key.lower() or 'info' in key.lower():
                            metadata['file_metadata'] = data[key].item() if data[key].ndim == 0 else data[key].tolist()
                            break
            except:
                pass
        
        return metadata
    
    def bulk_process_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.npy",
        channel_names: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Process all EEG files in a directory.
        
        Args:
            directory_path: Path to the directory
            file_pattern: Pattern for matching files
            channel_names: List of channel names to use for all files
            user_id: ID of the user who owns the data
            recursive: Whether to process subdirectories recursively
            
        Returns:
            Processing results
        """
        import glob
        
        # Find files matching the pattern
        if recursive:
            search_pattern = os.path.join(directory_path, "**", file_pattern)
            files = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(directory_path, file_pattern)
            files = glob.glob(search_pattern)
        
        results = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'file_results': {}
        }
        
        # Process each file
        for file_path in files:
            file_extension = os.path.splitext(file_path)[1]
            file_format = 'numpy'
            
            if file_extension.lower() in ['.edf']:
                file_format = 'edf'
            elif file_extension.lower() in ['.gdf']:
                file_format = 'gdf'
            
            try:
                file_result = self.process_eeg_file(
                    file_path=file_path,
                    file_format=file_format,
                    channel_names=channel_names,
                    recording_id=os.path.basename(file_path),
                    user_id=user_id,
                    metadata={'source_directory': directory_path}
                )
                
                results['file_results'][file_path] = {
                    'success': 'error' not in file_result,
                    'tokens': file_result.get('tokenization', {}).get(self.tokenizer_type, {}).get('count', 0)
                }
                
                if 'error' not in file_result:
                    results['processed'] += 1
                else:
                    results['failed'] += 1
                    if self.debug:
                        print(f"Error processing file {file_path}: {file_result['error']}")
                
            except Exception as e:
                results['failed'] += 1
                results['file_results'][file_path] = {
                    'success': False,
                    'error': str(e)
                }
                if self.debug:
                    print(f"Exception processing file {file_path}: {e}")
        
        return results 

    def _store_tokens(
        self, 
        tokens: List[Dict[str, Any]], 
        recording_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store tokens in the vector database.
        
        Args:
            tokens: List of tokens to store
            recording_id: Identifier for the recording
            metadata: Additional metadata
            
        Returns:
            Storage results
        """
        if not self.vector_store:
            return {'success': False, 'count': 0, 'error': 'Vector store not provided'}
        
        # Initialize results
        results = {
            'success': False,
            'count': 0,
            'tokenizer_results': {}
        }
        
        # Group tokens by tokenizer type
        tokenizer_types = {}
        for token in tokens:
            tokenizer_type = token.get('tokenizer_type', 'unknown')
            if tokenizer_type not in tokenizer_types:
                tokenizer_types[tokenizer_type] = []
            tokenizer_types[tokenizer_type].append(token)
        
        # Target embedding dimension (from the vector_store)
        target_dim = getattr(self.vector_store, 'embedding_dimension', 512)
        
        # Store tokens by type
        total_stored = 0
        successful_types = []
        
        for tokenizer_type, type_tokens in tokenizer_types.items():
            embeddings = []
            token_texts = []
            
            # Get the appropriate tokenizer
            tokenizer = None
            if tokenizer_type == 'feature':
                tokenizer = self.feature_tokenizer
            elif tokenizer_type == 'frequency':
                tokenizer = self.frequency_tokenizer
            elif tokenizer_type == 'autoencoder':
                tokenizer = self.autoencoder_tokenizer
            
            if not tokenizer:
                results['tokenizer_results'][tokenizer_type] = {
                    'success': False,
                    'count': 0,
                    'error': f'Tokenizer not available for type {tokenizer_type}'
                }
                continue
            
            # Generate embeddings and token texts
            for token in type_tokens:
                embedding = tokenizer.token_to_embedding(token)
                
                # Pad or truncate embedding to match the target dimension
                if embedding.shape[0] < target_dim:
                    # Pad with zeros if smaller than target
                    padded_embedding = np.zeros(target_dim)
                    padded_embedding[:embedding.shape[0]] = embedding
                    embedding = padded_embedding
                elif embedding.shape[0] > target_dim:
                    # Truncate if larger than target
                    embedding = embedding[:target_dim]
                
                token_text = tokenizer.decode_token(token)
                
                embeddings.append(embedding)
                token_texts.append(token_text)
                
                if self.debug and token == type_tokens[0]:
                    print(f"Original embedding dimension: {tokenizer.token_to_embedding(token).shape[0]}")
                    print(f"Padded embedding dimension: {embedding.shape[0]}")
            
            # Update metadata for each token
            token_metadata = [
                {**metadata, 'tokenizer_type': tokenizer_type} for _ in type_tokens
            ]
            
            # Store in vector database
            storage_result = self.vector_store.store_batch_embeddings(
                embeddings=embeddings,
                token_data_list=type_tokens,
                token_text_list=token_texts,
                metadata_list=token_metadata,
                recording_id=recording_id
            )
            
            # Update results
            results['tokenizer_results'][tokenizer_type] = storage_result
            
            if storage_result.get('success', False):
                successful_types.append(tokenizer_type)
                total_stored += storage_result.get('count', 0)
        
        # Update summary results
        results['success'] = len(successful_types) > 0
        results['count'] = total_stored
        
        if self.debug:
            print(f"Stored {total_stored} tokens in the vector database")
            for tokenizer_type, type_result in results['tokenizer_results'].items():
                success = type_result.get('success', False)
                count = type_result.get('count', 0)
                print(f"  {tokenizer_type}: {'✓' if success else '✗'} {count} tokens")
        
        return results