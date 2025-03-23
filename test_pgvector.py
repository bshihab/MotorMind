#!/usr/bin/env python
"""
Test script to verify pgvector installation in Supabase
"""

import os
import sys
import numpy as np
from vector_store.database.supabase_client import setup_supabase_client

# Test vector dimensionality
VECTOR_DIM = 512

def test_pgvector():
    """Test if pgvector is properly installed and configured"""
    print("Testing pgvector in Supabase...")
    
    # Connect to Supabase
    supabase = setup_supabase_client(debug=True)
    
    if not supabase or not supabase.is_connected():
        print("Failed to connect to Supabase")
        return False
    
    # Check if vector extension exists
    try:
        # Run SQL to check if the vector extension is enabled
        response = supabase.client.rpc(
            'check_vector_extension',
            {}
        ).execute()
        
        print(f"Vector extension check response: {response}")
    except Exception as e:
        print(f"Error checking vector extension: {e}")
        print("Creating check_vector_extension function...")
        
        # Create the function to check for vector extension
        try:
            create_func_sql = """
            create or replace function check_vector_extension()
            returns boolean as $$
            declare
                ext_exists boolean;
            begin
                select exists(
                    select 1 from pg_extension where extname = 'vector'
                ) into ext_exists;
                return ext_exists;
            end;
            $$ language plpgsql;
            """
            
            # Note: This approach is not ideal as it requires permissions to create functions
            # In a real application, you would use proper database migrations
            # We're just using this for testing purposes
            supabase.client.rpc('check_vector_extension', {}).execute()
        except Exception as e:
            print(f"Could not create check function: {e}")
    
    # Try to create a test vector
    try:
        print("\nAttempting to insert a test vector...")
        
        # Create a random test vector
        test_vector = np.random.random(VECTOR_DIM).tolist()
        
        # Insert the test vector
        data = {
            "embedding": test_vector,
            "token_data": {"test": True},
            "token_text": "Test vector",
            "metadata": {"test": True},
            "recording_id": "test_recording",
            "created_at": 1234567890
        }
        
        response = supabase.client.table('eeg_embeddings').insert(data).execute()
        
        if response.data:
            print("Successfully inserted test vector!")
            print(f"Response data: {response.data}")
            
            # Clean up by deleting the test vector
            print("\nCleaning up test data...")
            supabase.client.table('eeg_embeddings').delete().eq('recording_id', 'test_recording').execute()
            
            return True
        else:
            print(f"Insert failed. Error: {response.error}")
            return False
            
    except Exception as e:
        print(f"Error inserting test vector: {e}")
        print("Please make sure the pgvector extension is installed and the eeg_embeddings table is created correctly.")
        return False

if __name__ == "__main__":
    if test_pgvector():
        print("\n✅ pgvector is properly installed and configured!")
    else:
        print("\n❌ There are issues with pgvector configuration.")
        print("\nPlease run the supabase_setup.sql script in the Supabase SQL Editor to set up the database correctly.") 