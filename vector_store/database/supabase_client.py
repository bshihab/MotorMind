"""
Supabase Client

This module provides a simple wrapper for the Supabase client with vector database functionality.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Union

try:
    from supabase import create_client, Client
    from postgrest.exceptions import APIError
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Supabase packages not found. Install with: pip install supabase")


class SupabaseWrapper:
    """A wrapper for the Supabase client that handles common operations and error handling."""
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the Supabase wrapper.
        
        Args:
            supabase_url: Supabase project URL (if None, looks for SUPABASE_URL env var)
            supabase_key: Supabase API key (if None, looks for SUPABASE_KEY env var)
            debug: Whether to print debug information
        """
        self.debug = debug
        self.client = None
        
        if not SUPABASE_AVAILABLE:
            if self.debug:
                print("Supabase packages not available. Vector storage will not work.")
            return
        
        # Get Supabase credentials
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            if self.debug:
                print("Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY env vars.")
            return
        
        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            if self.debug:
                print("Connected to Supabase successfully")
        except Exception as e:
            if self.debug:
                print(f"Failed to connect to Supabase: {e}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if the client is connected to Supabase."""
        return self.client is not None
    
    def run_rpc(
        self,
        function_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a remote procedure call on Supabase.
        
        Args:
            function_name: Name of the RPC function
            params: Parameters to pass to the function
            
        Returns:
            Response from the RPC function
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Supabase")
        
        try:
            start_time = time.time()
            response = self.client.rpc(function_name, params).execute()
            
            if self.debug:
                print(f"RPC {function_name} took {time.time() - start_time:.2f}s")
            
            if response.data is not None:
                return {"data": response.data, "error": None}
            else:
                return {"data": None, "error": response.error}
        except Exception as e:
            if self.debug:
                print(f"RPC {function_name} failed: {e}")
            return {"data": None, "error": str(e)}
    
    def ensure_table_exists(
        self,
        table_name: str,
        schema: List[Dict[str, Any]]
    ) -> bool:
        """
        Ensure a table exists with the given schema.
        
        Args:
            table_name: Name of the table
            schema: Schema definition for the table
            
        Returns:
            True if the table exists or was created successfully
        """
        if not self.is_connected():
            return False
        
        try:
            # Check if table exists by fetching a single row
            response = self.client.table(table_name).select("*").limit(1).execute()
            
            # If no error, table exists
            if response.error is None:
                if self.debug:
                    print(f"Table {table_name} already exists")
                return True
            
            # Table doesn't exist, create it
            # Note: This is a simplified example - in production you would use actual
            # SQL migrations through Supabase's management interface or API
            
            # For demonstration purposes only - actual implementation would require SQL execution privileges
            if self.debug:
                print(f"Table {table_name} doesn't exist, would need to create it")
                print(f"Schema: {json.dumps(schema, indent=2)}")
            
            return False
        
        except Exception as e:
            if self.debug:
                print(f"Error checking table {table_name}: {e}")
            return False
    
    def insert(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        upsert: bool = False
    ) -> Dict[str, Any]:
        """
        Insert data into a table.
        
        Args:
            table_name: Name of the table
            data: Data to insert (dict or list of dicts)
            upsert: Whether to upsert (update if exists)
            
        Returns:
            Response from the insert operation
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Supabase")
        
        try:
            start_time = time.time()
            
            if upsert:
                response = self.client.table(table_name).upsert(data).execute()
            else:
                response = self.client.table(table_name).insert(data).execute()
            
            if self.debug:
                print(f"Insert into {table_name} took {time.time() - start_time:.2f}s")
                if isinstance(data, list):
                    print(f"Inserted {len(data)} rows")
                else:
                    print("Inserted 1 row")
            
            return {"data": response.data, "error": response.error}
        
        except Exception as e:
            if self.debug:
                print(f"Insert into {table_name} failed: {e}")
            return {"data": None, "error": str(e)}
    
    def select(
        self,
        table_name: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        order: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Select data from a table.
        
        Args:
            table_name: Name of the table
            columns: Columns to select
            filters: Filters to apply
            order: Order by clause
            limit: Maximum number of rows to return
            
        Returns:
            Response from the select operation
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Supabase")
        
        try:
            start_time = time.time()
            
            query = self.client.table(table_name).select(columns)
            
            # Apply filters
            if filters:
                for column, value in filters.items():
                    # Handle special operators
                    if isinstance(value, dict) and len(value) == 1:
                        operator, val = list(value.items())[0]
                        if operator == "eq":
                            query = query.eq(column, val)
                        elif operator == "neq":
                            query = query.neq(column, val)
                        elif operator == "gt":
                            query = query.gt(column, val)
                        elif operator == "lt":
                            query = query.lt(column, val)
                        elif operator == "gte":
                            query = query.gte(column, val)
                        elif operator == "lte":
                            query = query.lte(column, val)
                        elif operator == "like":
                            query = query.like(column, val)
                        elif operator == "ilike":
                            query = query.ilike(column, val)
                        elif operator == "in":
                            query = query.in_(column, val)
                    else:
                        # Default to equality
                        query = query.eq(column, value)
            
            # Apply order
            if order:
                if order.startswith("-"):
                    query = query.order(order[1:], desc=True)
                else:
                    query = query.order(order)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            if self.debug:
                print(f"Select from {table_name} took {time.time() - start_time:.2f}s")
                if response.data:
                    print(f"Selected {len(response.data)} rows")
            
            return {"data": response.data, "error": response.error}
        
        except Exception as e:
            if self.debug:
                print(f"Select from {table_name} failed: {e}")
            return {"data": None, "error": str(e)}
    
    def update(
        self,
        table_name: str,
        data: Dict[str, Any],
        match: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update data in a table.
        
        Args:
            table_name: Name of the table
            data: Data to update
            match: Match criteria for rows to update
            
        Returns:
            Response from the update operation
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Supabase")
        
        try:
            start_time = time.time()
            
            query = self.client.table(table_name).update(data)
            
            # Apply match criteria
            for column, value in match.items():
                query = query.eq(column, value)
            
            response = query.execute()
            
            if self.debug:
                print(f"Update {table_name} took {time.time() - start_time:.2f}s")
                if response.data:
                    print(f"Updated {len(response.data)} rows")
            
            return {"data": response.data, "error": response.error}
        
        except Exception as e:
            if self.debug:
                print(f"Update {table_name} failed: {e}")
            return {"data": None, "error": str(e)}
    
    def delete(
        self,
        table_name: str,
        match: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete data from a table.
        
        Args:
            table_name: Name of the table
            match: Match criteria for rows to delete
            
        Returns:
            Response from the delete operation
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Supabase")
        
        try:
            start_time = time.time()
            
            query = self.client.table(table_name).delete()
            
            # Apply match criteria
            for column, value in match.items():
                query = query.eq(column, value)
            
            response = query.execute()
            
            if self.debug:
                print(f"Delete from {table_name} took {time.time() - start_time:.2f}s")
                if response.data:
                    print(f"Deleted {len(response.data)} rows")
            
            return {"data": response.data, "error": response.error}
        
        except Exception as e:
            if self.debug:
                print(f"Delete from {table_name} failed: {e}")
            return {"data": None, "error": str(e)}
            

def setup_supabase_client(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    debug: bool = False
) -> Optional[SupabaseWrapper]:
    """
    Set up a Supabase client.
    
    Args:
        supabase_url: Supabase project URL (if None, looks for SUPABASE_URL env var or file)
        supabase_key: Supabase API key (if None, looks for SUPABASE_KEY env var or file)
        debug: Whether to print debug information
        
    Returns:
        SupabaseWrapper instance or None if connection failed
    """
    try:
        # Try to load URL from file if not provided and not in env
        if not supabase_url and "SUPABASE_URL" not in os.environ:
            url_file_path = os.path.join(os.getcwd(), "supabase_url.txt")
            if os.path.exists(url_file_path):
                with open(url_file_path, "r") as f:
                    supabase_url = f.read().strip()
                    if debug:
                        print(f"Loaded Supabase URL from {url_file_path}")
        
        # Try to load key from file if not provided and not in env
        if not supabase_key and "SUPABASE_KEY" not in os.environ:
            key_file_path = os.path.join(os.getcwd(), "supabase_key.txt")
            if os.path.exists(key_file_path):
                with open(key_file_path, "r") as f:
                    supabase_key = f.read().strip()
                    if debug:
                        print(f"Loaded Supabase key from {key_file_path}")
        
        # Load from .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            env_file_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_file_path):
                load_dotenv(env_file_path)
                if debug:
                    print(f"Loaded environment variables from {env_file_path}")
        except ImportError:
            if debug:
                print("python-dotenv not available. Install with: pip install python-dotenv")
        
        # Create the client
        client = SupabaseWrapper(supabase_url, supabase_key, debug)
        if client.is_connected():
            return client
        return None
    except Exception as e:
        if debug:
            print(f"Failed to set up Supabase client: {e}")
        return None 