"""文档解析 -> 切块 -> 向量化 -> 入库 的异步任务。"""
from __future__ import annotations

import os

from src.cache.redis_client import invalidate_cache
from src.config.settings import get_settings
from src.db.connection import get_pool
from src.document_parser.loader import load_text
from src.document_parser.semantic_splitter import split_text
from src.embedding.base import get_embedding
from src.vector_store.base import Chunk, get_vector_store


async def process_document(ctx: dict, document_id: str, department: str | None = None, secret_level: int = 0) -> None:
    """后台任务：解析 -> 结构化切块 -> 向量化 -> 原子写库 -> 更新状态。"""
    settings = get_settings()
    pool = await get_pool()

    row = await pool.fetchrow(
        "SELECT id, filename, version FROM documents WHERE id = $1::uuid", document_id
    )
    if row is None:
        return

    await pool.execute(
        "UPDATE documents SET status='processing', updated_at=now() WHERE id=$1::uuid", document_id
    )
    try:
        file_path = os.path.join(settings.upload_dir_abs, row["filename"])
        text = load_text(file_path)

        chunks_text = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks_text:
            raise ValueError("解析后无有效文本")

        embedding = get_embedding()
        vectors = await embedding.embed(chunks_text)

        version = row["version"]
        chunk_objs = [
            Chunk(
                document_id=document_id,
                chunk_index=i,
                content=t,
                source_file=row["filename"],
                document_version=version,
                department=department,
                secret_level=secret_level,
            )
            for i, t in enumerate(chunks_text)
        ]

        store = get_vector_store()
        await store.replace_document(document_id, chunk_objs, vectors)

        await pool.execute(
            "UPDATE documents SET status='completed', chunk_count=$2, updated_at=now() WHERE id=$1::uuid",
            document_id,
            len(chunks_text),
        )
        await invalidate_cache()
    except Exception as e:  # noqa: BLE001
        await pool.execute(
            "UPDATE documents SET status='failed', error=$2, updated_at=now() WHERE id=$1::uuid",
            document_id,
            str(e),
        )
        raise  # 交给 arq 重试；超 max_tries 后 job 判失败，状态已标 failed（死信兜底）
