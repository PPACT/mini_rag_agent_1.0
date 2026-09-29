"""Embedding 抽象层。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """嵌入模型统一接口。上层业务只依赖此接口，切换实现类不改业务代码。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本 -> 向量列表（维度与 settings.embedding_dim 一致）。"""
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放底层资源（默认无操作；持有连接/会话的实现应覆盖）。

        ⚠️ 单例化之后必须留这个口子：长生命周期的连接不显式关闭就是资源泄漏
        （D9-⑧）。由应用 shutdown（`src/main.py`）调用。
        """
        return None


_instance: BaseEmbedding | None = None


def get_embedding() -> BaseEmbedding:
    """返回当前 embedding 实现（**单例**）。默认本地 Ollama；后期切云端只改这里。

    ⚠️ D9-⑧：原来是**每次调用都 new 一个** `OllamaEmbedding`，而它内部又
    `async with httpx.AsyncClient()` —— 于是每个请求都新建客户端、**完全没有连接复用**。
    对比 `get_reranker()` 是单例且注释解释了为什么必须单例，这里当时漏了。

    ⚠️ 与 reranker 的单例**差别**：本实现的单例持有长生命周期 `AsyncClient`，
    因此 shutdown 必须调 `close_embedding()`（reranker 持有的是模型，无需关闭）。
    """
    global _instance
    if _instance is None:
        from src.embedding.ollama_embedding import OllamaEmbedding

        _instance = OllamaEmbedding()
    return _instance


async def close_embedding() -> None:
    """关闭 embedding 单例并释放其 HTTP 连接池（应用 shutdown 调用）。"""
    global _instance
    if _instance is not None:
        await _instance.aclose()
        _instance = None
