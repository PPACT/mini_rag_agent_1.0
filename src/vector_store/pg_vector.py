"""PgVectorStore：pgvector + 全文检索实现（asyncpg）。"""
from __future__ import annotations

from src.db.connection import get_pool
from src.document_parser.tokenizer import tokenize
from src.vector_store.base import AccessFilter, Chunk, VectorStore

_SELECT_COLS = """
    c.id::text, c.document_id::text, c.chunk_index, c.content,
    c.source_file, c.department, c.secret_level,
    c.start_offset, c.end_offset, c.title,
    c.page, c.raw_table, c.table_complex
"""

# 向量距离的原生运算符形态（$1 = 查询向量）。
# ⚠️ **必须原样出现在 ORDER BY 里**，HNSW 索引才会被使用——见 `_build_search_sql` 的 D9-⑪ 注释。
_VEC_DIST = "c.embedding <=> $1::vector"


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
        page=r["page"],
        raw_table=r["raw_table"],
        table_complex=r["table_complex"],
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

        返回 (hard_conds, weight_expr, weight_params, params, n, penalized)：
        - hard_conds: 必须硬过滤的 WHERE 条件（数据状态、密级；**软过滤时不含部门**）
        - weight_expr: score 的权重表达式（软过滤时 CASE 降权范围外；否则 '1.0'）
        - weight_params: 软过滤时 [departments, penalty]；否则 []
        - params / n: 其余参数与计数器
        - penalized: 本次是否真的在降权 —— `_build_search_sql` 靠它决定**排序能否用原生运算符**
          （D9-⑪：weight 恒为 1.0 时不必写成表达式，写了反而让 HNSW 索引失效）
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
            penalized = True
        else:
            if filters.departments:
                hard.append(f"c.department = ANY(${n})")
                params.append(filters.departments)
                n += 1
            weight_expr = "1.0"
            weight_params = []
            penalized = False
        return hard, weight_expr, weight_params, params, n, penalized

    @classmethod
    def _build_search_sql(cls, vec_str: str, filters: AccessFilter, top_k: int,
                          soft_scope: bool | None = None, penalty: float | None = None) -> tuple[str, list]:
        """构建向量检索 SQL 与参数（纯函数，便于单测 AccessFilter→SQL 翻译）。

        ⚠️ **D9-⑪：排序形态决定 HNSW 索引是否生效。**

        pgvector 的 HNSW 索引**只在 `ORDER BY <向量列> <=> <查询向量>` 这种直接用
        运算符的形态下生效**。写成 `ORDER BY <表达式>`（哪怕表达式只是给距离乘 1.0）
        会退化成 **Seq Scan + Sort** —— 1231 切片下两种写法都是几毫秒，看不出差别，
        **数据一涨就是数量级差异**（本项目语料会持续增长）。

        故这里分两种形态，**数值完全一致、只是排序写法不同**：

        - 无软过滤（weight 恒为 `1.0`）：`ORDER BY c.embedding <=> $1`（距离升序 == score 降序）→ 走索引
        - 软过滤（真的在降权）：只能 `ORDER BY score DESC`，**索引用不上**（软过滤本就是退路）

        回归保护：`tests/test_pg_vector.py` 钉住"无软过滤时不得出现 `ORDER BY score DESC`"；
        计划形态实测见 `eval/verify_hnsw_plan.py`。
        """
        from src.config.settings import get_settings

        s = get_settings()
        if soft_scope is None:
            soft_scope = s.scope_soft_enabled
        if penalty is None:
            penalty = s.scope_soft_penalty
        hard, weight, wparams, params, n, penalized = cls._filter_parts(
            filters, 2, soft_scope, penalty)
        if penalized:
            score_expr = f"(1 - ({_VEC_DIST})) * ({weight})"
            order_by = "score DESC"
        else:
            score_expr = f"1 - ({_VEC_DIST})"
            order_by = _VEC_DIST
        sql = f"""
            SELECT {_SELECT_COLS},
                   {score_expr} AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {' AND '.join(hard)}
            ORDER BY {order_by}
            LIMIT ${n}
        """
        return sql, [vec_str, *params, *wparams, top_k]

    @classmethod
    def _build_lexical_sql(cls, tsquery: str, filters: AccessFilter, top_k: int,
                           soft_scope: bool | None = None, penalty: float | None = None) -> tuple[str, list]:
        """构建词法检索 SQL 与参数（$1 为 tsquery 字符串，权限过滤与向量侧共用）。

        排序按 `ts_rank`，**没有"索引友好的运算符形态"这回事**（GIN 索引只用
        于 `@@` 匹配、不用于排序），故这里保持 `ORDER BY score DESC`——
        与向量侧 D9-⑪ 的取舍不同，不是因为漏改。
        """
        from src.config.settings import get_settings

        s = get_settings()
        if soft_scope is None:
            soft_scope = s.scope_soft_enabled
        if penalty is None:
            penalty = s.scope_soft_penalty
        hard, weight, wparams, params, n, _penalized = cls._filter_parts(
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
                                            start_offset, end_offset, title,
                                            page, raw_table, table_complex, content_tsv,
                                            content_hash, table_id, image_path)
                        VALUES ($1::uuid, $2, $3, $4, $5::vector, $6, $7, $8, $9, $10, $11,
                                $12, $13, $14, to_tsvector('simple', $15), $16, $17, $18)
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
                        chunk.page,
                        chunk.raw_table,
                        chunk.table_complex,
                        # 词法索引吃两份：自然语言版 + 原表（2.0-1 口径）。
                        # 向量侧仍只算 content（embedding 在调用方按 content 算好传进来）。
                        #
                        # ⚠️ **但别把它的作用说大了**（2026-09-30 实测）：
                        # 因为自然语言版**逐字保留**所有值（那正是 2.0-1 的保真验收），
                        # `tokenize(content + raw_table)` 与 `tokenize(content)`
                        # 的 **token 集合完全相同** —— 实测 `raw_table` 独有的 token 数为 **0**。
                        # 所以它**不是**"新增了一条精确命中路径"（那样写是错的），真实作用是：
                        #   ① **词频提升**：ts_rank 计次 → 表格块在数值类查询下排名略微靠前；
                        #   ② **保险**：万一将来自然语言版变得有损（丢列/丢行），
                        #      原表还在词法索引里，不至于连值都搜不到。
                        # ⭐ **"本字段的独有 token 应为空"是一个可观测的健康指标**（文档侧建议）：
                        #    **不为空 → 自然语言版有损（丢了列/值）→ 去查 `render_nl`，别查索引。**
                        #    守卫：`tests/test_tables.py::test_raw_table_adds_no_unique_tokens`
                        #    （带反向用例 —— 用一份有损的渲染验证它真能失败）。
                        #
                        # 中文需 Python 侧分词后再交给 tsvector。
                        tokenize(chunk.content + ("\n" + chunk.raw_table if chunk.raw_table else "")),
                        # `2.0-30`：内容哈希（只重嵌变了的块）+ 两个预留列
                        chunk.content_hash,
                        chunk.table_id,
                        chunk.image_path,
                    )

    async def delete_by_document(self, document_id: str) -> None:
        """物理删除某文档的全部向量（软删由 documents.is_deleted 负责）。"""
        pool = await get_pool(self._kb)
        await pool.execute("DELETE FROM chunks WHERE document_id = $1::uuid", document_id)

    async def count(self) -> int:
        pool = await get_pool(self._kb)
        return int(await pool.fetchval("SELECT count(*) FROM chunks"))
