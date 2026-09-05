"""RAG 检索：query 向量化 + 向量检索 + 权限过滤。"""
from __future__ import annotations

from src.config.settings import get_settings
from src.embedding.base import get_embedding
from src.vector_store.base import AccessFilter, Chunk, get_vector_store


def format_context(chunks: list[Chunk]) -> str:
    """把命中切片拼成带 [来源N] 标记的上下文。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[来源{i}] (文件:{c.source_file}, 块:{c.chunk_index})\n{c.content}")
    return "\n\n".join(parts)


async def retrieve(
    question: str,
    departments: list[str] | None,
    secret_level: int | None,
) -> tuple[str, list[Chunk]]:
    """检索：返回 (上下文文本, 命中切片列表)。"""
    settings = get_settings()

    embedding = get_embedding()
    q_vec = (await embedding.embed([question]))[0]

    store = get_vector_store()
    filters = AccessFilter(departments=departments, secret_level_le=secret_level)
    chunks = await store.search(q_vec, filters, settings.top_k)

    return format_context(chunks), chunks
