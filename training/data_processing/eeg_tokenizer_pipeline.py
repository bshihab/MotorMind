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
from vector_store.database.supabase_vector_store import SupabaseVectorStore


class EEGTokenizerPipeline:
    """
    Pipeline for processing EEG data, tokenizing it, and preparing it for
    storage in the vector database.
    """
    
    def __init__(
        self,
        tokenizer_type: str = 'feature',  # 'feature', 'frequency', or 'both'
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
        self.vector_store = vector_store
        self.fs = fs
        self.window_size = window_size
        self.window_shift = window_shift
        self.debug = debug
        
        # Default parameters for tokenizers
        self.tokenizer_params = tokenizer_params or {}
        
        # Initialize tokenizers
        self._init_tokenizers()
    
    def _init_tokenizers(self) -> None:
        """Initialize the tokenizers based on the selected type."""
        params = {
            'fs': self.fs,
            'window_size': self.window_size,
            'window_shift': self.window_shift,
            **self.tokenizer_params
        }
        
        if self.tokenizer_type == 'feature' or self.tokenizer_type == 'both':
            self.feature_tokenizer = FeatureTokenizer(**params)
        
        if self.tokenizer_type == 'frequency' or self.tokenizer_type == 'both':
            self.frequency_tokenizer = FrequencyTokenizer(**params)
    
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
            recording_id: ID of the recording
            user_id: ID of the user
            metadata: Additional metadata
            
        Returns:
            Processing results
        """
        # Default channel names if not provided
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(eeg_data.shape[0])]
        
        # Initialize results
        results = {
            'tokenization': {},
            'embeddings': {},
            'storage': {'success': False, 'count': 0}
        }
        
        # Tokenize data
        if self.tokenizer_type == 'feature' or self.tokenizer_type == 'both':
            feature_tokens = self.feature_tokenizer.tokenize(eeg_data, channel_names)
            results['tokenization']['feature'] = {
                'count': len(feature_tokens),
                'tokens': feature_tokens
            }
        
        if self.tokenizer_type == 'frequency' or self.tokenizer_type == 'both':
            frequency_tokens = self.frequency_tokenizer.tokenize(eeg_data, channel_names)
            results['tokenization']['frequency'] = {
                'count': len(frequency_tokens),
                'tokens': frequency_tokens
            }
        
        # Generate embeddings
        if self.tokenizer_type == 'feature' or self.tokenizer_type == 'both':
            feature_embeddings = []
            feature_token_texts = []
            
            for token in feature_tokens:
                embedding = self.feature_tokenizer.token_to_embedding(token)
                token_text = self.feature_tokenizer.decode_token(token)
                
                feature_embeddings.append(embedding)
                feature_token_texts.append(token_text)
            
            results['embeddings']['feature'] = {
                'count': len(feature_embeddings),
                'embeddings': feature_embeddings,
                'token_texts': feature_token_texts
            }
        
        if self.tokenizer_type == 'frequency' or self.tokenizer_type == 'both':
            frequency_embeddings = []
            frequency_token_texts = []
            
            for token in frequency_tokens:
                embedding = self.frequency_tokenizer.token_to_embedding(token)
                token_text = self.frequency_tokenizer.decode_token(token)
                
                frequency_embeddings.append(embedding)
                frequency_token_texts.append(token_text)
            
            results['embeddings']['frequency'] = {
                'count': len(frequency_embeddings),
                'embeddings': frequency_embeddings,
                'token_texts': frequency_token_texts
            }
        
        # Store embeddings if vector store is provided
        if self.vector_store is not None:
            if self.tokenizer_type == 'feature' or self.tokenizer_type == 'both':
                feature_storage = self.vector_store.store_batch_embeddings(
                    embeddings=feature_embeddings,
                    token_data_list=feature_tokens,
                    token_text_list=feature_token_texts,
                    metadata_list=[{'tokenizer_type': 'feature', **(metadata or {})} for _ in feature_tokens],
                    recording_id=recording_id,
                    user_id=user_id
                )
                results['storage']['feature'] = feature_storage
            
            if self.tokenizer_type == 'frequency' or self.tokenizer_type == 'both':
                frequency_storage = self.vector_store.store_batch_embeddings(
                    embeddings=frequency_embeddings,
                    token_data_list=frequency_tokens,
                    token_text_list=frequency_token_texts,
                    metadata_list=[{'tokenizer_type': 'frequency', **(metadata or {})} for _ in frequency_tokens],
                    recording_id=recording_id,
                    user_id=user_id
                )
                results['storage']['frequency'] = frequency_storage
            
            # Calculate total success
            results['storage']['success'] = all([
                results['storage'].get(t, {}).get('success', False) 
                for t in results['storage'] if t != 'success' and t != 'count'
            ])
            
            # Calculate total count
            results['storage']['count'] = sum([
                results['storage'].get(t, {}).get('count', 0) 
                for t in results['storage'] if t != 'success' and t != 'count'
            ])
        
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