#!/usr/bin/env python
"""
Test script to diagnose issues with the match_embeddings function in Supabase
"""

import os
import sys
import numpy as np
import json
from dotenv import load_dotenv
from vector_store.database.supabase_client import setup_supabase_client

# Test vector dimensionality
VECTOR_DIM = 512

def test_match_embeddings():
    """Test the match_embeddings RPC function"""
    print("Testing match_embeddings RPC function in Supabase...")
    
    # Load environment variables
    load_dotenv()
    
    # Connect to Supabase
    supabase = setup_supabase_client(debug=True)
    
    if not supabase or not hasattr(supabase, 'client'):
        print("Failed to connect to Supabase")
        return False
    
    # Create a random test vector
    test_vector = np.random.random(VECTOR_DIM).tolist()
    
    # Test parameters to match what's in our code
    table_name = "eeg_embeddings"
    match_count = 5
    match_threshold = 0.6
    
    # Attempt to call match_embeddings directly
    print("\nAttempting to call match_embeddings RPC function...")
    
    try:
        payload = {
            "query_embedding": test_vector,
            "match_threshold": match_threshold,
            "match_count": match_count,
            "table_name": table_name
        }
        
        print(f"Request payload size: {len(json.dumps(payload))} characters")
        print(f"First few elements of embedding: {test_vector[:5]}...")
        
        response = supabase.client.rpc(
            "match_embeddings",
            payload
        ).execute()
        
        # Check the response
        if response.data is not None:
            print(f"Success! match_embeddings returned {len(response.data)} results")
            if len(response.data) > 0:
                print(f"First result: {response.data[0]}")
            else:
                print("No matching results found (which is expected for a random vector)")
            return True
        else:
            print(f"Failed. Error: {response.error}")
            return False
            
    except Exception as e:
        print(f"Error calling match_embeddings: {str(e)}")
        
        # Try to extract more details from the error message
        error_str = str(e)
        if "details" in error_str:
            import re
            details_match = re.search(r"'details': '([^']*)'", error_str)
            if details_match:
                print(f"Error details: {details_match.group(1)}")
        
        # Test a simpler approach - just get the first few records
        print("\nTrying fallback approach...")
        try:
            fallback_response = supabase.client.table(table_name).select("*").limit(5).execute()
            print(f"Fallback succeeded, returned {len(fallback_response.data)} records")
            
            # If we can retrieve records but not query them, it suggests the RPC function is the issue
            if fallback_response.data:
                print("The table exists and contains records, but match_embeddings RPC is failing")
                print("Check if the function definition matches the SQL from supabase_setup.sql")
        except Exception as fallback_error:
            print(f"Fallback error: {fallback_error}")
        
        return False

if __name__ == "__main__":
    if test_match_embeddings():
        print("\n✅ match_embeddings function works correctly!")
    else:
        print("\n❌ There are issues with the match_embeddings function.")
        print("\nPlease run the supabase_setup.sql script in the Supabase SQL Editor to verify the RPC function.") 