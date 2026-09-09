"""本地 Ollama embedding 实现（直连 /api/embed，CUDA）。"""
from __future__ import annotations

import httpx

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """通过本地 Ollama 服务生成向量（模型 bge-m3，版本锁定不自动更新）。

    注意：绕开 LiteLLM（其 Ollama embedding 路径不稳定），直连 /api/embed。
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._dim = settings.embedding_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        async with httpx.AsyncClient(timeout=180.0) as client:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                result.extend(await self._embed_batch(client, batch))
        return result

    async def _embed_batch(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        resp = await client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if embeddings and len(embeddings[0]) != self._dim:
            raise ValueError(
                f"embedding 维度不匹配：模型返回 {len(embeddings[0])} 维，配置 EMBEDDING_DIM={self._dim}"
            )
        return embeddings
