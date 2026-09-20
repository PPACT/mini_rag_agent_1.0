"""PgVectorStore：pgvector + 全文检索实现（asyncpg）。"""
from __future__ import annotations

from src.db.connection import get_pool
from src.document_parser.tokenizer import tokenize
from src.vector_store.base import AccessFilter, Chunk, VectorStore

_SELECT_COLS = """
    c.id::text, c.document_id::text, c.chunk_index, c.content,
    c.source_file, c.department, c.secret_level,
    c.start_offset, c.end_offset, c.title
"""


def _row_to_chunk(r) -> Chunk:
    return Chunk(
        id=r["id"],
        document_id=r["document_id"],
        chunk_index=r["chunk_index"],
        content=r["content"],
        source_file=r["source_file"],
        department=r["department"],
        secret_level=r["secret_level"],
        start_offset=r["start_offset"],
        end_offset=r["end_offset"],
        title=r["title"],
        score=float(r["score"]),
    )


class PgVectorStore(VectorStore):
    """基于 PostgreSQL + pgvector 的向量库实现。

    - 向量检索：embedding 以字符串字面量 + `::vector` 传入（asyncpg 无原生 vector 类型）
    - 词法检索：`content_tsv`（Python 侧 jieba 分词后写入）+ ts_rank
    - **两库分离**：实例绑定一个 kb（真实 / 压测），所有查询走该库的连接池
    """

    def __init__(self, kb: str) -> None:
        """绑定知识库。

        ⚠️ `kb` **必填**——见 `src/db/kb.py`：给默认值会让"忘传"静默落到真实库。
        """
        from src.db.kb import validate

        self._kb = validate(kb)

    @staticmethod
    def _vec_str(embedding: list[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    @staticmethod
    def _filter_parts(filters: AccessFilter, start_n: int, soft_scope: bool, penalty: float):
        """构造过滤片段（向量检索与词法检索**共用**，确保两侧语义一致）。

        返回 (hard_conds, weight_expr, weight_params, params, n)：
        - hard_conds: 必须硬过滤的 WHERE 条件（数据状态、密级；**软过滤时不含部门**）
        - weight_expr: score 的权重表达式（软过滤时 CASE 降权范围外；否则 '1.0'）
        - weight_params: 软过滤时 [departments, penalty]；否则 []
        - params / n: 其余参数与计数器
        """
        params: list = []
        n = start_n
        hard = ["d.is_deleted = false", "c.is_deprecated = false"]
        if filters.secret_level_le is not None:
            hard.append(f"c.secret_level <= ${n}")
            params.append(filters.secret_level_le)
            n += 1

        if filters.departments and soft_scope:
            # 软过滤：范围外不排除，降权排序
            d_idx, p_idx = n, n + 1
            n += 2
            weight_expr = (f"CASE WHEN c.department = ANY(${d_idx}) "
                           f"THEN 1.0 ELSE ${p_idx}::double precision END")
            weight_params = [filters.departments, penalty]
        else:
            if filters.departments:
                hard.append(f"c.department = ANY(${n})")
                params.append(filters.departments)
                n += 1
            weight_expr = "1.0"
            weight_params = []
        return hard, weight_expr, weight_params, params, n

    @classmethod
    def _build_search_sql(cls, vec_str: str, filters: AccessFilter, top_k: int,
                          soft_scope: bool | None = None, penalty: float | None = None) -> tuple[str, list]:
        """构建向量检索 SQL 与参数（纯函数，便于单测 AccessFilter→SQL 翻译）。"""
        from src.config.settings import get_settings

        s = get_settings()
        if soft_scope is None:
            soft_scope = s.scope_soft_enabled
        if penalty is None:
            penalty = s.scope_soft_penalty
        hard, weight, wparams, params, n = cls._filter_parts(
            filters, 2, soft_scope, penalty)
        sql = f"""
            SELECT {_SELECT_COLS},
                   (1 - (c.embedding <=> $1::vector)) * ({weight}) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {' AND '.join(hard)}
            ORDER BY score DESC
            LIMIT ${n}
        """
        return sql, [vec_str, *params, *wparams, top_k]

    @classmethod
    def _build_lexical_sql(cls, tsquery: str, filters: AccessFilter, top_k: int,
                           soft_scope: bool | None = None, penalty: float | None = None) -> tuple[str, list]:
        """构建词法检索 SQL 与参数（$1 为 tsquery 字符串，权限过滤与向量侧共用）。"""
        from src.config.settings import get_settings

        s = get_settings()
        if soft_scope is None:
            soft_scope = s.scope_soft_enabled
        if penalty is None:
            penalty = s.scope_soft_penalty
        hard, weight, wparams, params, n = cls._filter_parts(
            filters, 2, soft_scope, penalty)
        hard.append("c.content_tsv @@ to_tsquery('simple', $1)")
        sql = f"""
            SELECT {_SELECT_COLS},
                   ts_rank(c.content_tsv, to_tsquery('simple', $1)) * ({weight}) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {' AND '.join(hard)}
            ORDER BY score DESC
            LIMIT ${n}
        """
        return sql, [tsquery, *params, *wparams, top_k]

    async def search(self, embedding: list[float], filters: AccessFilter, top_k: int) -> list[Chunk]:
        pool = await get_pool(self._kb)
        sql, params = self._build_search_sql(self._vec_str(embedding), filters, top_k)
        rows = await pool.fetch(sql, *params)
        return [_row_to_chunk(r) for r in rows]

    async def search_lexical(self, query: str, filters: AccessFilter, top_k: int) -> list[Chunk]:
        """全文检索（BM25-ish）：命中精确词（缩写、编号、数字）。

        score 是 ts_rank，与向量侧的余弦相似度**量纲不同**；
        上层用 RRF 按「排名」融合，因此量纲差异不影响结果。
        """
        tsquery = " | ".join(tokenize(query).split())  # OR 语义，靠 ts_rank 排序
        if not tsquery:
            return []
        pool = await get_pool(self._kb)
        sql, params = self._build_lexical_sql(tsquery, filters, top_k)
        rows = await pool.fetch(sql, *params)
        return [_row_to_chunk(r) for r in rows]

    async def replace_document(self, document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """同一事务内：先删旧 chunk，再插新 chunk。"""
        pool = await get_pool(self._kb)
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks WHERE document_id = $1::uuid", document_id)
                for chunk, emb in zip(chunks, embeddings):
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, document_version,
                                            content, embedding, department, secret_level, source_file,
                                            start_offset, end_offset, title, content_tsv)
                        VALUES ($1::uuid, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11,
                                to_tsvector('simple', $12))
                        """,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.document_version,
                        chunk.content,
                        self._vec_str(emb),
                        chunk.department,
                        chunk.secret_level,
                        chunk.source_file,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.title,
                        tokenize(chunk.content),  # 中文需 Python 侧分词后再交给 tsvector
                    )

    async def delete_by_document(self, document_id: str) -> None:
        """物理删除某文档的全部向量（软删由 documents.is_deleted 负责）。"""
        pool = await get_pool(self._kb)
        await pool.execute("DELETE FROM chunks WHERE document_id = $1::uuid", document_id)

    async def count(self) -> int:
        pool = await get_pool(self._kb)
        return int(await pool.fetchval("SELECT count(*) FROM chunks"))
