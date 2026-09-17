"""RAG 检索：多查询扩展 → 向量粗排 → RRF 融合 → 精排(Rerank) → 权限过滤。"""
from __future__ import annotations

from src.config.settings import get_settings
from src.embedding.base import get_embedding
from src.rag.query_rewriter import expand_queries
from src.rag.reranker import get_reranker
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
    use_rerank: bool | None = None,
    use_hybrid: bool | None = None,
    top_k: int | None = None,
) -> tuple[str, list[Chunk]]:
    """混合检索 + 两段式：词法/向量多路粗排 → RRF 融合 → 精排 → 返回 (上下文文本, 命中切片)。

    各开关 None 时读配置；显式传 False 可跑基线（评测对比用）。
    """
    settings = get_settings()
    if use_rewrite is None:
        use_rewrite = settings.query_rewrite_enabled
    if use_rerank is None:
        use_rerank = settings.rerank_enabled
    if use_hybrid is None:
        use_hybrid = settings.hybrid_search_enabled
    if top_k is None:
        top_k = settings.top_k

    # 粗排召回数：开启精排则放大召回，给精排留出挑选空间
    recall_k = max(top_k, settings.rerank_candidates) if use_rerank else top_k

    queries = [question]
    if use_rewrite:
        queries = await expand_queries(question, settings.query_rewrite_count)

    embedding = get_embedding()
    query_vectors = await embedding.embed(queries)

    store = get_vector_store()
    filters = AccessFilter(departments=departments, secret_level_le=secret_level)

    ranked_lists: list[list[Chunk]] = []
    vector_lists: list[list[Chunk]] = []

    # 词法侧（混合检索）：兜底"精确词"查询（缩写、编号、型号）
    # 不支持的实现返回空列表，自动退化为纯向量检索
    if use_hybrid:
        lexical = await store.search_lexical(question, filters, recall_k)
        if lexical:
            ranked_lists.append(lexical)

    # 向量侧（多查询扩展的每一路）
    for vec in query_vectors:
        lst = await store.search(vec, filters, recall_k)
        vector_lists.append(lst)
        ranked_lists.append(lst)

    if len(ranked_lists) == 1:
        candidates = ranked_lists[0]
    else:
        candidates = rrf_merge(ranked_lists, settings.rrf_k)[:recall_k]

    # ⚠️ 修复 score 语义：RRF 用 setdefault 保留"第一出现"那路的原始分，
    # 而词法路是 ts_rank（0~0.1）、向量路是余弦相似度（0~1），两者不可比。
    # 这里对每个候选回填**向量路的真实相似度**（若存在），使 score 语义统一、可做阈值判断。
    vec_best: dict[str, float] = {}
    for lst in vector_lists:
        for c in lst:
            if c.id and c.score > vec_best.get(c.id, -1.0):
                vec_best[c.id] = c.score
    for c in candidates:
        if c.id in vec_best:
            c.score = vec_best[c.id]

    # 精排：用更强的判断力纠正"语义相近但答非所问"
    if use_rerank and len(candidates) > top_k:
        chunks = await get_reranker().rerank(question, candidates, top_k)
    else:
        chunks = candidates[:top_k]

    return format_context(chunks), chunks
