-- =============================================================
-- RAG-Demo 数据库初始化：documents 主表 + chunks 向量表
-- 执行方式：psql <DATABASE_URL> -f src/db/init_db.sql
-- =============================================================

-- pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 文档主表（生命周期 + 软删 + 版本，为异步队列/增量同步预留）
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
);

-- 向量切片表（每条 chunk 绑定文档 id / 序号 / 版本，便于精准清理）
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
);

-- 向量检索索引（HNSW，余弦距离）
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
-- 按文档查询 / 清理
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
-- 权限过滤（department + secret_level）
CREATE INDEX IF NOT EXISTS idx_chunks_perm ON chunks (department, secret_level);
-- 软删过滤 / 增量同步
CREATE INDEX IF NOT EXISTS idx_documents_is_deleted ON documents (is_deleted);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents (updated_at);
