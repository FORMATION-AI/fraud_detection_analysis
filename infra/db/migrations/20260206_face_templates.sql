-- Face templates for pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS face_templates (
    template_id uuid PRIMARY KEY,
    user_id text NOT NULL,
    embedding vector(512) NOT NULL,
    is_mean_template boolean NOT NULL DEFAULT false,
    model_name text NOT NULL,
    model_version text NOT NULL,
    quality_json jsonb,
    active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS face_templates_user_id_idx ON face_templates (user_id);
CREATE INDEX IF NOT EXISTS face_templates_active_idx ON face_templates (active);

-- Optional: cosine similarity with ivfflat (requires ANALYZE; adjust lists for scale)
CREATE INDEX IF NOT EXISTS face_templates_embedding_ivfflat
    ON face_templates USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
