"""
Supabase Vector Store Implementation

This module implements a vector database using Supabase for storing and retrieving
EEG token embeddings for the RAG system.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import time

class SupabaseVectorStore:
    """
    Vector store implementation using Supabase for storing and retrieving EEG token embeddings.
    """
    
    def __init__(
        self,
        supabase_client,
        embedding_dimension: int = 512,
        table_name: str = "eeg_embeddings",
        debug: bool = False
    ):
        """
        Initialize the Supabase vector store.
        
        Args:
            supabase_client: Initialized Supabase client
            embedding_dimension: Dimension of the embedding vectors
            table_name: Name of the table to store embeddings
            debug: Whether to print debug information
        """
        self.supabase = supabase_client
        self.embedding_dimension = embedding_dimension
        self.table_name = table_name
        self.debug = debug
        
        # Initialize the table if it doesn't exist
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """Ensure that the vector store table exists in Supabase."""
        if not self.supabase or not hasattr(self.supabase, 'client'):
            if self.debug:
                print("Supabase client not properly initialized")
            return
        
        try:
            # Check if the table exists by attempting a simple query
            self.supabase.client.table(self.table_name).select("count", count="exact").limit(1).execute()
            if self.debug:
                print(f"Table '{self.table_name}' exists")
        except Exception as e:
            if self.debug:
                print(f"Error checking if table exists: {e}")
                print("You may need to create the vector store table manually with pgvector extension")
    
    def store_embedding(
        self,
        embedding: np.ndarray,
        token_data: Dict[str, Any],
        token_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store an embedding vector with associated token data.
        
        Args:
            embedding: Embedding vector as numpy array
            token_data: Original token data dictionary
            token_text: Text representation of the token for LLM processing
            metadata: Additional metadata
            recording_id: ID of the associated EEG recording
            user_id: ID of the user who owns this data
            
        Returns:
            Response from the database
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Convert embedding to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Prepare data for insertion
            data = {
                "embedding": embedding_list,
                "token_data": token_data,
                "token_text": token_text,
                "metadata": metadata or {},
                "recording_id": recording_id,
                "user_id": user_id,
                "created_at": time.time()
            }
            
            # Insert data
            response = self.supabase.client.table(self.table_name).insert(data).execute()
            
            return {
                "success": True,
                "data": response.data[0] if response.data else None
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error storing embedding: {e}")
            return {"error": str(e), "success": False}
    
    def store_batch_embeddings(
        self,
        embeddings: List[np.ndarray],
        token_data_list: List[Dict[str, Any]],
        token_text_list: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store multiple embeddings in a batch operation.
        
        Args:
            embeddings: List of embedding vectors
            token_data_list: List of token data dictionaries
            token_text_list: List of text representations
            metadata_list: List of metadata dictionaries (optional)
            recording_id: ID of the associated EEG recording
            user_id: ID of the user who owns this data
            
        Returns:
            Response from the database
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        if len(embeddings) != len(token_data_list) or len(embeddings) != len(token_text_list):
            return {"error": "Mismatched list lengths", "success": False}
        
        # Use empty metadata if not provided
        if metadata_list is None:
            metadata_list = [{} for _ in range(len(embeddings))]
        
        try:
            # Prepare batch data
            batch_data = []
            for i, embedding in enumerate(embeddings):
                batch_data.append({
                    "embedding": embedding.tolist(),
                    "token_data": token_data_list[i],
                    "token_text": token_text_list[i],
                    "metadata": metadata_list[i],
                    "recording_id": recording_id,
                    "user_id": user_id,
                    "created_at": time.time()
                })
            
            # Insert batch data
            response = self.supabase.client.table(self.table_name).insert(batch_data).execute()
            
            return {
                "success": True,
                "count": len(response.data) if response.data else 0
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error storing batch embeddings: {e}")
            return {"error": str(e), "success": False}
    
    def query_similar(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        similarity_threshold: float = 0.6,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            filter_criteria: Additional filter criteria
            
        Returns:
            Query results
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Convert embedding to list for JSON serialization
            if isinstance(query_embedding, np.ndarray):
                # Sanitize the embedding: replace NaN/Inf with 0
                query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0)
                embedding_list = query_embedding.tolist()
            else:
                embedding_list = query_embedding
            
            # Debug output
            if self.debug:
                print(f"Querying for similar embeddings (dim={len(embedding_list)}, threshold={similarity_threshold})")
            
            # Call the match_embeddings function
            try:
                payload = {
                    "query_embedding": embedding_list,
                    "match_threshold": similarity_threshold,
                    "match_count": limit,
                    "table_name": self.table_name
                }
                
                response = self.supabase.client.rpc(
                    "match_embeddings",
                    payload
                ).execute()
                
                results = response.data if response.data else []
                
                if self.debug:
                    print(f"Found {len(results)} similar tokens")
                
                # Apply additional filters if provided
                if filter_criteria and results:
                    filtered_results = []
                    for result in results:
                        match = True
                        for key, value in filter_criteria.items():
                            if key not in result or result[key] != value:
                                match = False
                                break
                        if match:
                            filtered_results.append(result)
                    results = filtered_results
                
                return {
                    "success": True,
                    "results": results
                }
                
            except Exception as match_error:
                if self.debug:
                    print(f"Error calling match_embeddings: {match_error}")
                
                # Fall through to fallback method
                raise match_error
                
        except Exception as e:
            if self.debug:
                print(f"Falling back to simple retrieval: {e}")
            
            # Fallback to a simpler approach - find recent tokens without vector search
            try:
                # Just get some recent tokens as fallback
                fallback_response = self.supabase.client.table(self.table_name).select("*").limit(limit).execute()
                
                if self.debug:
                    print(f"Fallback retrieved {len(fallback_response.data if fallback_response.data else [])} tokens")
                
                return {
                    "success": True,
                    "results": fallback_response.data if fallback_response.data else [],
                    "fallback": True
                }
            except Exception as fallback_error:
                if self.debug:
                    print(f"Fallback error: {fallback_error}")
                
                return {
                    "success": False,
                    "error": str(e),
                    "results": []
                }
    
    def delete_embeddings(
        self,
        filter_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete embeddings based on filter criteria.
        
        Args:
            filter_criteria: Filter criteria for deletion
            
        Returns:
            Response from the database
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        if not filter_criteria:
            return {"error": "No filter criteria provided", "success": False}
        
        try:
            # Build the query with the filter criteria
            query = self.supabase.client.table(self.table_name)
            
            for key, value in filter_criteria.items():
                query = query.eq(key, value)
            
            # Execute the delete operation
            response = query.delete().execute()
            
            return {
                "success": True,
                "count": len(response.data) if response.data else 0
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error deleting embeddings: {e}")
            return {"error": str(e), "success": False}
    
    def get_token_by_id(self, token_id: str) -> Dict[str, Any]:
        """
        Retrieve a token by its ID.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Token data
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Query for the token
            response = self.supabase.client.table(self.table_name).select("*").eq("id", token_id).execute()
            
            if response.data and len(response.data) > 0:
                return {
                    "success": True,
                    "data": response.data[0]
                }
            else:
                return {
                    "success": False,
                    "error": "Token not found"
                }
            
        except Exception as e:
            if self.debug:
                print(f"Error retrieving token: {e}")
            return {"error": str(e), "success": False}
    
    def get_tokens_by_recording(self, recording_id: str) -> Dict[str, Any]:
        """
        Retrieve all tokens associated with a recording.
        
        Args:
            recording_id: ID of the recording
            
        Returns:
            List of tokens
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False, "tokens": []}
        
        try:
            # Query for tokens by recording ID
            response = self.supabase.client.table(self.table_name).select("*").eq("recording_id", recording_id).execute()
            
            return {
                "success": True,
                "tokens": response.data if response.data else []
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error retrieving tokens by recording: {e}")
            return {"error": str(e), "success": False, "tokens": []}
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embeddings in the vector store.
        
        Returns:
            Statistics about the embeddings
        """
        if not self.supabase or not hasattr(self.supabase, 'client'):
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get total count
            count_response = self.supabase.client.table(self.table_name).select("count", count="exact").execute()
            total_count = count_response.count if hasattr(count_response, 'count') else 0
            
            # Get count by recording ID
            recording_stats_query = f"""
            SELECT recording_id, COUNT(*) as token_count
            FROM {self.table_name}
            GROUP BY recording_id
            ORDER BY token_count DESC
            LIMIT 10
            """
            
            recording_stats_response = self.supabase.client.rpc("run_query", {"query": recording_stats_query}).execute()
            
            return {
                "success": True,
                "total_count": total_count,
                "recording_stats": recording_stats_response.data if recording_stats_response.data else []
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error getting embedding stats: {e}")
            return {"error": str(e), "success": False} 