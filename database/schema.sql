-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store legal documents metadata and full text content
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    document_number VARCHAR(100),
    title TEXT NOT NULL,
    url TEXT,
    legal_type VARCHAR(100),
    legal_sectors TEXT,
    issuing_authority VARCHAR(255),
    issuance_date VARCHAR(50),
    signers TEXT,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Table to store chunks of documents for Vector RAG search
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768), -- Default to 768 dimensions for Vietnamese sentence-transformers / bkai-bi-encoder
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now())
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_document_number ON documents(document_number);
CREATE INDEX IF NOT EXISTS idx_documents_legal_type ON documents(legal_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

-- GIN index for full-text search on document contents (useful for hybrid search)
CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents USING gin(to_tsvector('simple', content));
CREATE INDEX IF NOT EXISTS idx_document_chunks_fts ON document_chunks USING gin(to_tsvector('simple', content));
