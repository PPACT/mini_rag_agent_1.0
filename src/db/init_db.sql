-- =============================================================
-- RAG-Demo 数据库初始化：documents 主表 + chunks 向量表
--
-- ⚠️⚠️ **本文件已过期，请勿执行**（2026-09-29 标注）⚠️⚠️
--
-- 它只是**最初的 schema 快照**（`alembic/versions/0001_create_tables.py` 与它等价）。
-- 此后 0002 / 0003 / 0004 三个迁移又加了 **6 个列**，本文件**都没有**：
--   chunks.start_offset / chunks.end_offset / chunks.title
--   chunks.is_deprecated / chunks.content_tsv
--   documents.original_name
--
-- 照本文件建表 → **schema 是残的**，且**建表时不会报错**，
-- 直到查询才失败（缺列）—— 又是一次"静默故障"。
--
-- ✅ **建表请用统一入口**（自动对两个库各跑一次 + 校验 schema 一致）：
--        python scripts/migrate_all.py
--    （原因见 `scripts/migrate_all.py` 的模块说明：迁移只跑一个库不会报错，
--      但两库 schema 会不一致，后续查询莫名失败。）
--
-- 本文件保留仅为「最初长什么样」的存档。
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
