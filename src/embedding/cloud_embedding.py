"""云端 Embedding 占位实现（后期阿里云 / 字节 / OpenAI 等）。"""
from __future__ import annotations

from src.embedding.base import BaseEmbedding


class CloudEmbedding(BaseEmbedding):
    """占位：后期接入云端 embedding 服务时实现，业务代码零改动。"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("云端 Embedding 尚未接入，仅占位")
