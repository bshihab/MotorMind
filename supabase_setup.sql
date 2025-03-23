-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table for EEG embeddings
CREATE TABLE IF NOT EXISTS eeg_embeddings (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(512),  -- Default embedding size of 512
    token_data JSONB,       -- Original token data as JSON
    token_text TEXT,        -- Text representation of the token
    metadata JSONB,         -- Additional metadata
    recording_id TEXT,      -- ID of the EEG recording
    user_id TEXT,           -- User ID (can be null)
    created_at DOUBLE PRECISION  -- Creation timestamp
);

-- Create an index for similarity search
CREATE INDEX IF NOT EXISTS embedding_vector_idx ON eeg_embeddings USING ivfflat (embedding vector_l2_ops);

-- Create a stored procedure for similarity search
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(512),
    match_threshold FLOAT,
    match_count INT,
    table_name TEXT
)
RETURNS TABLE (
    id BIGINT,
    embedding vector(512),
    token_data JSONB,
    token_text TEXT,
    metadata JSONB,
    recording_id TEXT,
    user_id TEXT,
    created_at DOUBLE PRECISION,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY EXECUTE format('
        SELECT 
            id, 
            embedding, 
            token_data, 
            token_text, 
            metadata, 
            recording_id, 
            user_id,
            created_at, 
            1 - (embedding <-> $1) AS similarity
        FROM %I
        WHERE 1 - (embedding <-> $1) > $2
        ORDER BY similarity DESC
        LIMIT $3
    ', table_name)
    USING query_embedding, match_threshold, match_count;
END;
$$; 