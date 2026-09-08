"""创建 documents + chunks 表（初始 schema，等价于原 init_db.sql）。

Revision ID: 0001
Revises: (无)
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename    TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'deleted')),
            version     INTEGER NOT NULL DEFAULT 1,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            error       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            is_deleted  BOOLEAN NOT NULL DEFAULT false
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index      INTEGER NOT NULL,
            document_version INTEGER NOT NULL DEFAULT 1,
            content          TEXT NOT NULL,
            embedding        vector(1024),
            department       TEXT,
            secret_level     INTEGER NOT NULL DEFAULT 0,
            source_file      TEXT,
            create_time      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index, document_version)
        )
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_perm ON chunks (department, secret_level)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_is_deleted ON documents (is_deleted)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents (updated_at)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
