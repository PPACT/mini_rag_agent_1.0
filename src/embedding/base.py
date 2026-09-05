"""Embedding 抽象层。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """嵌入模型统一接口。上层业务只依赖此接口，切换实现类不改业务代码。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 -> 向量列表（维度与 settings.embedding_dim 一致）。"""
        raise NotImplementedError


def get_embedding() -> BaseEmbedding:
    """返回当前 embedding 实现。默认本地 Ollama；后期切云端只改这里。"""
    from src.embedding.ollama_embedding import OllamaEmbedding

    return OllamaEmbedding()
