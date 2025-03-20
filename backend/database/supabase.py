"""
Supabase Client Configuration and Database Utilities

This module provides functions for interacting with the Supabase backend
for the MotorMind EEG-LLM project.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import required for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from supabase import Client, PostgrestResponse


class SupabaseClient:
    """
    Wrapper class for Supabase client with utility methods for EEG data management.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (if None, read from env)
            key: Supabase API key (if None, read from env)
            debug: Whether to enable debug logging
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self.debug = debug
        self.client = None
        
        if not self.url or not self.key:
            if self.debug:
                print("Warning: Supabase URL or key not provided")
        else:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Supabase client with the provided credentials."""
        try:
            from supabase import create_client
            self.client = create_client(self.url, self.key)
            if self.debug:
                print("Supabase client initialized successfully")
        except ImportError:
            if self.debug:
                print("Error: 'supabase' package not installed. Run: pip install supabase")
        except Exception as e:
            if self.debug:
                print(f"Error initializing Supabase client: {e}")
    
    def is_connected(self) -> bool:
        """Check if the Supabase client is connected and working."""
        if not self.client:
            return False
        
        try:
            # Try a simple query to check connection
            self.client.table("eeg_datasets").select("count", count="exact").limit(1).execute()
            return True
        except Exception:
            return False

    # ----- Dataset Management -----
    
    def create_dataset(
        self,
        name: str,
        description: str = "",
        metadata: Dict[str, Any] = None,
        is_public: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new EEG dataset.
        
        Args:
            name: Dataset name
            description: Dataset description
            metadata: Additional metadata as a dictionary
            is_public: Whether the dataset is publicly accessible
            user_id: User ID (if None, use authenticated user)
            
        Returns:
            Created dataset record or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Create dataset record
            data = {
                "name": name,
                "description": description,
                "metadata": metadata or {},
                "is_public": is_public,
                "user_id": user_id
            }
            
            response = self.client.table("eeg_datasets").insert(data).execute()
            
            return {
                "success": True,
                "data": response.data[0] if response.data else None
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error creating dataset: {e}")
            return {"error": str(e), "success": False}
    
    def get_datasets(
        self,
        user_id: Optional[str] = None,
        include_public: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get datasets for a user.
        
        Args:
            user_id: User ID (if None, use authenticated user)
            include_public: Whether to include public datasets
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of datasets or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Build query
            query = self.client.table("eeg_datasets").select("*")
            
            if include_public:
                query = query.or_(f"user_id.eq.{user_id},is_public.eq.true")
            else:
                query = query.eq("user_id", user_id)
            
            response = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "data": response.data,
                "count": len(response.data)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error retrieving datasets: {e}")
            return {"error": str(e), "success": False}
    
    # ----- EEG Recording Management -----
    
    def create_recording(
        self,
        dataset_id: str,
        recording_date: str,
        duration: int,
        channels: int,
        sampling_rate: int,
        metadata: Dict[str, Any] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new EEG recording entry.
        
        Args:
            dataset_id: ID of the dataset this recording belongs to
            recording_date: Date/time of the recording (ISO format)
            duration: Duration in seconds
            channels: Number of channels
            sampling_rate: Sampling rate in Hz
            metadata: Additional metadata as a dictionary
            user_id: User ID (if None, use authenticated user)
            
        Returns:
            Created recording record or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Create recording record
            data = {
                "dataset_id": dataset_id,
                "user_id": user_id,
                "recording_date": recording_date,
                "duration": duration,
                "channels": channels,
                "sampling_rate": sampling_rate,
                "metadata": metadata or {}
            }
            
            response = self.client.table("eeg_recordings").insert(data).execute()
            
            return {
                "success": True,
                "data": response.data[0] if response.data else None
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error creating recording: {e}")
            return {"error": str(e), "success": False}
    
    def get_recordings(
        self,
        dataset_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get EEG recordings.
        
        Args:
            dataset_id: Filter by dataset ID (if None, return all accessible recordings)
            user_id: Filter by user ID (if None, use authenticated user)
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of recordings or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Build query
            query = self.client.table("eeg_recordings").select("*")
            
            if dataset_id:
                query = query.eq("dataset_id", dataset_id)
            
            query = query.eq("user_id", user_id)
            
            response = query.order("recording_date", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "data": response.data,
                "count": len(response.data)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error retrieving recordings: {e}")
            return {"error": str(e), "success": False}
    
    # ----- EEG Data Management -----
    
    def store_eeg_data(
        self,
        recording_id: str,
        timestamps: List[float],
        channel_data: Dict[int, List[float]]
    ) -> Dict[str, Any]:
        """
        Store EEG data in the database.
        
        Args:
            recording_id: ID of the recording
            timestamps: List of timestamps (in seconds)
            channel_data: Dictionary mapping channel number to data values
            
        Returns:
            Result of the operation
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Prepare data for insertion
            data_to_insert = []
            
            for i, timestamp in enumerate(timestamps):
                for channel, values in channel_data.items():
                    if i < len(values):
                        data_to_insert.append({
                            "recording_id": recording_id,
                            "time": timestamp,
                            "channel": channel,
                            "value": values[i]
                        })
            
            # Insert in batches to avoid exceeding size limits
            batch_size = 1000
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i+batch_size]
                self.client.table("eeg_data").insert(batch).execute()
            
            return {
                "success": True,
                "count": len(data_to_insert)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error storing EEG data: {e}")
            return {"error": str(e), "success": False}
    
    def store_eeg_features(
        self,
        recording_id: str,
        features: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Store extracted EEG features in the database.
        
        Args:
            recording_id: ID of the recording
            features: List of feature dictionaries
            
        Returns:
            Result of the operation
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Prepare data for insertion
            data_to_insert = []
            
            for feature_dict in features:
                data_to_insert.append({
                    "recording_id": recording_id,
                    "window_start": feature_dict.get("window_start"),
                    "window_end": feature_dict.get("window_end"),
                    "feature_data": json.dumps(feature_dict) if isinstance(feature_dict, dict) else feature_dict
                })
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i+batch_size]
                self.client.table("eeg_features").insert(batch).execute()
            
            return {
                "success": True,
                "count": len(data_to_insert)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error storing EEG features: {e}")
            return {"error": str(e), "success": False}
    
    # ----- Model Predictions -----
    
    def get_predictions(
        self,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get model predictions.
        
        Args:
            recording_id: Filter by recording ID
            user_id: Filter by user ID (if None, use authenticated user)
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of predictions or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Build query
            query = self.client.table("model_predictions").select("*")
            
            if recording_id:
                query = query.eq("recording_id", recording_id)
            
            query = query.eq("user_id", user_id)
            
            response = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
            
            return {
                "success": True,
                "data": response.data,
                "count": len(response.data)
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error retrieving predictions: {e}")
            return {"error": str(e), "success": False}
    
    # ----- Research Collaboration -----
    
    def create_project(
        self,
        name: str,
        description: str = "",
        is_public: bool = False,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new research project.
        
        Args:
            name: Project name
            description: Project description
            is_public: Whether the project is publicly accessible
            user_id: User ID (if None, use authenticated user)
            
        Returns:
            Created project record or error
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Get current user if not specified
            if not user_id:
                user = self.client.auth.get_user()
                user_id = user.user.id
            
            # Create project record
            data = {
                "name": name,
                "description": description,
                "is_public": is_public,
                "owner_id": user_id
            }
            
            response = self.client.table("research_projects").insert(data).execute()
            
            # Add owner as a member
            if response.data:
                project_id = response.data[0]["id"]
                self.client.table("project_members").insert({
                    "project_id": project_id,
                    "user_id": user_id,
                    "role": "owner"
                }).execute()
            
            return {
                "success": True,
                "data": response.data[0] if response.data else None
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error creating project: {e}")
            return {"error": str(e), "success": False}
    
    def add_project_member(
        self,
        project_id: str,
        member_email: str,
        role: str = "viewer"
    ) -> Dict[str, Any]:
        """
        Add a member to a research project.
        
        Args:
            project_id: ID of the project
            member_email: Email of the user to add
            role: Role in the project (owner, editor, viewer)
            
        Returns:
            Result of the operation
        """
        if not self.client:
            return {"error": "Supabase client not initialized", "success": False}
        
        try:
            # Find user by email
            response = self.client.rpc(
                "get_user_by_email",
                {"email": member_email}
            ).execute()
            
            if not response.data:
                return {"error": "User not found", "success": False}
            
            member_id = response.data[0]["id"]
            
            # Add user to project
            self.client.table("project_members").insert({
                "project_id": project_id,
                "user_id": member_id,
                "role": role
            }).execute()
            
            return {
                "success": True
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error adding project member: {e}")
            return {"error": str(e), "success": False}


# Initialize the Singleton client
supabase = SupabaseClient()


# ----- Database Schema Creation -----

def create_database_schema(url: str, key: str) -> Dict[str, Any]:
    """
    Create the database schema for the MotorMind project.
    
    Args:
        url: Supabase project URL
        key: Supabase API key
        
    Returns:
        Result of the operation
    """
    try:
        from supabase import create_client
        client = create_client(url, key)
        
        # Execute SQL to create tables and extensions
        
        # Enable necessary extensions
        client.table("extensions").select("*").execute()  # This is just to check connection
        
        # Use raw SQL execution for extensions
        # TimescaleDB for time-series data
        client.rpc("exec_sql", {"query": "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"}).execute()
        
        # Create tables
        
        # 1. EEG Datasets
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS eeg_datasets (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID REFERENCES auth.users(id),
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB,
                is_public BOOLEAN DEFAULT FALSE
            );
            """
        }).execute()
        
        # 2. EEG Recordings
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS eeg_recordings (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                dataset_id UUID REFERENCES eeg_datasets(id),
                user_id UUID REFERENCES auth.users(id),
                recording_date TIMESTAMP WITH TIME ZONE,
                duration INTEGER,
                channels INTEGER,
                sampling_rate INTEGER,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        }).execute()
        
        # 3. EEG Data (time-series)
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS eeg_data (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                recording_id UUID REFERENCES eeg_recordings(id),
                channel INTEGER,
                value FLOAT,
                PRIMARY KEY (recording_id, time, channel)
            );
            """
        }).execute()
        
        # Convert to hypertable for TimescaleDB
        client.rpc("exec_sql", {
            "query": "SELECT create_hypertable('eeg_data', 'time', if_not_exists => TRUE);"
        }).execute()
        
        # 4. EEG Features
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS eeg_features (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                recording_id UUID REFERENCES eeg_recordings(id),
                window_start INTEGER,
                window_end INTEGER,
                feature_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        }).execute()
        
        # 5. Model Predictions
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                recording_id UUID REFERENCES eeg_recordings(id),
                user_id UUID REFERENCES auth.users(id),
                model_version TEXT NOT NULL,
                window_start INTEGER,
                window_end INTEGER,
                prediction TEXT,
                confidence FLOAT,
                explanation TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        }).execute()
        
        # 6. Research Projects
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS research_projects (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name TEXT NOT NULL,
                description TEXT,
                owner_id UUID REFERENCES auth.users(id),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_public BOOLEAN DEFAULT FALSE
            );
            """
        }).execute()
        
        # 7. Project Members
        client.rpc("exec_sql", {
            "query": """
            CREATE TABLE IF NOT EXISTS project_members (
                project_id UUID REFERENCES research_projects(id),
                user_id UUID REFERENCES auth.users(id),
                role TEXT NOT NULL,
                joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (project_id, user_id)
            );
            """
        }).execute()
        
        # Create function to look up users by email (for collaboration)
        client.rpc("exec_sql", {
            "query": """
            CREATE OR REPLACE FUNCTION get_user_by_email(email TEXT)
            RETURNS SETOF auth.users
            LANGUAGE sql
            SECURITY DEFINER
            SET search_path = public
            AS $$
                SELECT * FROM auth.users WHERE email = email;
            $$;
            """
        }).execute()
        
        # Set up RLS (Row Level Security) policies
        # This is simplified - in production you'd want more granular policies
        
        # Enable RLS on all tables
        for table in ["eeg_datasets", "eeg_recordings", "eeg_features", "model_predictions", 
                     "research_projects", "project_members"]:
            client.rpc("exec_sql", {
                "query": f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;"
            }).execute()
        
        # Create policies
        # Example for eeg_datasets
        client.rpc("exec_sql", {
            "query": """
            CREATE POLICY "Users can view their own datasets and public ones"
            ON eeg_datasets FOR SELECT
            USING (auth.uid() = user_id OR is_public = true);
            
            CREATE POLICY "Users can insert their own datasets"
            ON eeg_datasets FOR INSERT
            WITH CHECK (auth.uid() = user_id);
            
            CREATE POLICY "Users can update their own datasets"
            ON eeg_datasets FOR UPDATE
            USING (auth.uid() = user_id);
            """
        }).execute()
        
        return {
            "success": True,
            "message": "Database schema created successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 