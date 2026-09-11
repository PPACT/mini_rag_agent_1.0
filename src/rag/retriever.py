"""RAG 检索：多查询扩展 + 向量检索 + RRF 融合 + 权限过滤。"""
from __future__ import annotations

from src.config.settings import get_settings
from src.embedding.base import get_embedding
from src.rag.query_rewriter import expand_queries
from src.vector_store.base import AccessFilter, Chunk, get_vector_store


def format_context(chunks: list[Chunk]) -> str:
    """把命中切片拼成带 [来源N] 标记的上下文。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[来源{i}] (文件:{c.source_file}, 块:{c.chunk_index})\n{c.content}")
    return "\n\n".join(parts)


def rrf_merge(ranked_lists: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Reciprocal Rank Fusion：融合多路检索结果。

    score(chunk) = Σ 1/(k + rank)，按 chunk_id 去重，天然把"多路都命中"的块排前。
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for lst in ranked_lists:
        for rank, c in enumerate(lst, start=1):
            if c.id is None:
                continue
            scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(c.id, c)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [by_id[cid] for cid, _ in ordered]


async def retrieve(
    question: str,
    departments: list[str] | None,
    secret_level: int | None,
    *,
    use_rewrite: bool | None = None,
    top_k: int | None = None,
) -> tuple[str, list[Chunk]]:
    """检索：多查询扩展 → 各路向量检索 → RRF 融合 → 返回 (上下文文本, 命中切片)。

    use_rewrite=None 时读配置；显式传 False 可跑基线（评测对比用）。
    top_k=None 时读配置；评测可传更大值以观察完整排序。
    """
    settings = get_settings()
    if use_rewrite is None:
        use_rewrite = settings.query_rewrite_enabled
    if top_k is None:
        top_k = settings.top_k

    queries = [question]
    if use_rewrite:
        queries = await expand_queries(question, settings.query_rewrite_count)

    embedding = get_embedding()
    query_vectors = await embedding.embed(queries)

    store = get_vector_store()
    filters = AccessFilter(departments=departments, secret_level_le=secret_level)

    ranked_lists = [await store.search(vec, filters, top_k) for vec in query_vectors]

    if len(ranked_lists) == 1:
        chunks = ranked_lists[0]
    else:
        chunks = rrf_merge(ranked_lists, settings.rrf_k)[:top_k]

    return format_context(chunks), chunks
