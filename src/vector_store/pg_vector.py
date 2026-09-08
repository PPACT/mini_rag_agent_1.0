"""PgVectorStore：pgvector 实现（asyncpg）。"""
from __future__ import annotations

from src.db.connection import get_pool
from src.vector_store.base import AccessFilter, Chunk, VectorStore


class PgVectorStore(VectorStore):
    """基于 PostgreSQL + pgvector 的向量库实现。

    embedding 以字符串字面量 + `::vector` 传入（asyncpg 无原生 vector 类型）。
    """

    @staticmethod
    def _vec_str(embedding: list[float]) -> str:
        return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    @staticmethod
    def _build_search_sql(vec_str: str, filters: AccessFilter, top_k: int) -> tuple[str, list]:
        """构建检索 SQL 与参数（纯函数，便于单测 AccessFilter→SQL 翻译）。"""
        conds = ["d.is_deleted = false"]
        params: list = [vec_str]
        n = 2
        if filters.departments:
            conds.append(f"c.department = ANY(${n})")
            params.append(filters.departments)
            n += 1
        if filters.secret_level_le is not None:
            conds.append(f"c.secret_level <= ${n}")
            params.append(filters.secret_level_le)
            n += 1

        sql = f"""
            SELECT c.id::text, c.document_id::text, c.chunk_index, c.content,
                   c.source_file, c.department, c.secret_level,
                   (1 - (c.embedding <=> $1::vector)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {' AND '.join(conds)}
            ORDER BY c.embedding <=> $1::vector
            LIMIT ${n}
        """
        params.append(top_k)
        return sql, params

    async def search(self, embedding: list[float], filters: AccessFilter, top_k: int) -> list[Chunk]:
        pool = await get_pool()
        sql, params = self._build_search_sql(self._vec_str(embedding), filters, top_k)
        rows = await pool.fetch(sql, *params)
        return [
            Chunk(
                id=r["id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                source_file=r["source_file"],
                department=r["department"],
                secret_level=r["secret_level"],
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def replace_document(self, document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """同一事务内：先删旧 chunk，再插新 chunk。"""
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks WHERE document_id = $1::uuid", document_id)
                for chunk, emb in zip(chunks, embeddings):
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, chunk_index, document_version,
                                            content, embedding, department, secret_level, source_file)
                        VALUES ($1::uuid, $2, $3, $4, $5::vector, $6, $7, $8)
                        """,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.document_version,
                        chunk.content,
                        self._vec_str(emb),
                        chunk.department,
                        chunk.secret_level,
                        chunk.source_file,
                    )

    async def delete_by_document(self, document_id: str) -> None:
        """物理删除某文档的全部向量（软删由 documents.is_deleted 负责）。"""
        pool = await get_pool()
        await pool.execute("DELETE FROM chunks WHERE document_id = $1::uuid", document_id)

    async def count(self) -> int:
        pool = await get_pool()
        return int(await pool.fetchval("SELECT count(*) FROM chunks"))
