"""MilvusStore 占位桩（后期迁移 Milvus 时实现）。"""
from __future__ import annotations

from src.vector_store.base import AccessFilter, Chunk, VectorStore


class MilvusStore(VectorStore):
    """占位：后期迁移 Milvus 时实现，业务代码零改动（只改 VECTOR_STORE 配置）。"""

    def __init__(self, kb: str) -> None:
        """绑定知识库（真实 / 压测）。

        保留 kb：两库分离后，**真实库可以单独迁到 Milvus 而压测库留在 PG**
        ——这正是"双库可切换"的落地形态。
        """
        from src.db.kb import validate

        self._kb = validate(kb)

    async def search(self, embedding: list[float], filters: AccessFilter, top_k: int) -> list[Chunk]:
        raise NotImplementedError("Milvus 尚未接入，仅占位")

    async def replace_document(self, document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError("Milvus 尚未接入，仅占位")

    async def delete_by_document(self, document_id: str) -> None:
        raise NotImplementedError("Milvus 尚未接入，仅占位")

    async def count(self) -> int:
        raise NotImplementedError("Milvus 尚未接入，仅占位")
