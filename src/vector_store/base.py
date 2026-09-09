"""向量库抽象层 + 配置驱动工厂。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AccessFilter:
    """统一权限过滤条件（单一来源，各引擎各自翻译，保证过滤语义一致）。"""

    departments: list[str] | None = None   # 允许的部门列表（None = 不过滤）
    secret_level_le: int | None = None     # 密级上限（<=，None = 不过滤）


@dataclass
class Chunk:
    """检索/写入共用的切片结构（统一，不暴露引擎专有类型）。"""

    document_id: str
    chunk_index: int
    content: str
    department: str | None = None
    secret_level: int = 0
    source_file: str | None = None
    document_version: int = 1
    start_offset: int | None = None
    end_offset: int | None = None
    title: str | None = None
    is_deprecated: bool = False
    id: str | None = None          # 检索结果回填；写入时为空
    score: float = 0.0             # 相似度（检索结果）


class VectorStore(ABC):
    """向量库最小公共接口：search / replace_document / delete_by_document / count。

    Milvus 特有高级能力（混合检索、分区键等）不进此接口，走可选 mixin 或专属配置。
    """

    @abstractmethod
    async def search(self, embedding: list[float], filters: AccessFilter, top_k: int) -> list[Chunk]:
        """向量相似度检索 + 权限过滤，返回 top_k 切片。"""
        raise NotImplementedError

    @abstractmethod
    async def replace_document(self, document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """原子替换某文档的全部向量（同一事务：先删旧、再插新）。"""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None:
        """物理删除某文档全部向量。"""
        raise NotImplementedError

    @abstractmethod
    async def count(self) -> int:
        """切片总数（对账用）。"""
        raise NotImplementedError


def get_vector_store() -> VectorStore:
    """根据 settings.vector_store 返回实现（pgvector | milvus）。"""
    from src.config.settings import get_settings

    store_type = get_settings().vector_store.lower()
    if store_type == "pgvector":
        from src.vector_store.pg_vector import PgVectorStore

        return PgVectorStore()
    if store_type == "milvus":
        from src.vector_store.milvus import MilvusStore

        return MilvusStore()
    raise ValueError(f"未知向量库类型: {store_type}")
