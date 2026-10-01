-- init.sql — runs once on first Postgres boot (docker-entrypoint-initdb.d).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id        TEXT PRIMARY KEY,          -- payload id → ON CONFLICT target (idempotent)
  content   TEXT,
  embedding vector(384),               -- all-MiniLM-L6-v2 output dim (EMBED_MODE=real, the default).
                                       -- EMBED_MODE=hash emits 8 dims: use EGRESS=redis for plumbing
                                       -- runs, or change this to vector(8). The worker checks the
                                       -- declared dimension at boot and refuses to start on mismatch.
  meta      JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- HNSW index for downstream ANN search (build AFTER bulk load in real pipelines).
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
