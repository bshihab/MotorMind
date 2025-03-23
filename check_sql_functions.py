#!/usr/bin/env python
"""
Test script to check the SQL functions in Supabase
"""

import os
import sys
from vector_store.database.supabase_client import setup_supabase_client

def check_sql_functions():
    """Check if the required SQL functions exist in Supabase"""
    print("Checking SQL functions in Supabase...")
    
    # Connect to Supabase
    supabase = setup_supabase_client(debug=True)
    
    if not supabase or not supabase.is_connected():
        print("Failed to connect to Supabase")
        return False
    
    # Check if the function exists in pg_proc (system catalog)
    check_function_query = """
    SELECT proname, proargtypes, prosrc 
    FROM pg_catalog.pg_proc 
    WHERE proname IN ('match_embeddings', 'exec_sql')
    """
    
    try:
        # Try to execute a SQL query directly to check function existence
        response = supabase.client.from_("pg_catalog.pg_proc").select("proname").limit(1).execute()
        print(f"Direct catalog query response: {response}")
    except Exception as e:
        print(f"Error accessing pg_catalog directly: {e}")
    
    # Try to call match_embeddings directly
    try:
        print("\nTrying to call match_embeddings function...")
        test_vector = [0] * 512  # Create a zero vector of length 512
        
        response = supabase.client.rpc(
            "match_embeddings",
            {
                "query_embedding": test_vector,
                "match_threshold": 0.0,
                "match_count": 1,
                "table_name": "eeg_embeddings"
            }
        ).execute()
        
        print(f"Match embeddings response: {response}")
        return True
    except Exception as e:
        print(f"Error calling match_embeddings function: {e}")
    
    # Try to use exec_sql function
    try:
        print("\nTrying to call exec_sql function...")
        test_query = "SELECT COUNT(*) FROM eeg_embeddings"
        
        response = supabase.client.rpc(
            "exec_sql",
            {
                "sql": test_query
            }
        ).execute()
        
        print(f"Exec SQL response: {response}")
        return True
    except Exception as e:
        print(f"Error calling exec_sql function: {e}")
    
    return False

if __name__ == "__main__":
    if check_sql_functions():
        print("\n✅ Required SQL functions are available!")
    else:
        print("\n❌ Some required SQL functions are missing.")
        print("\nYou need to execute the SQL setup script:")
        print("\n1. Go to your Supabase project dashboard")
        print("2. Navigate to the SQL Editor")
        print("3. Execute the contents of supabase_setup.sql")
        print("4. Add the following SQL to create the exec_sql function:")
        print("""
    -- Create a function to execute arbitrary SQL (with proper permissions)
    CREATE OR REPLACE FUNCTION exec_sql(sql text)
    RETURNS JSONB
    LANGUAGE plpgsql
    SECURITY DEFINER
    AS $$
    DECLARE
        result JSONB;
    BEGIN
        EXECUTE sql INTO result;
        RETURN result;
    EXCEPTION WHEN OTHERS THEN
        RETURN jsonb_build_object('error', SQLERRM);
    END;
    $$;
    """) 