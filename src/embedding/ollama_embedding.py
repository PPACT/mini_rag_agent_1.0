"""本地 Ollama embedding 实现（直连 /api/embed，CUDA）。"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from src.config.settings import get_settings
from src.embedding.base import BaseEmbedding

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _is_loopback(url: str) -> bool:
    """base_url 是否指向本机。"""
    return (urlparse(url).hostname or "").lower() in _LOOPBACK_HOSTS


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
        self._keep_alive = settings.ollama_keep_alive
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """惰性创建并**复用**同一个客户端（D9-⑧）。

        原来 `embed()` 里 `async with httpx.AsyncClient(...)` 是**每次调用新建**，
        每请求一次都要重做 TCP+TLS 握手、且无法复用连接池。
        模型跑在 Ollama 端、本地进程无"模型重载"风险，故客户端可以长期持有。

        ⚠️ **本机服务绝不走代理**（`trust_env=False`）。httpx 默认 `trust_env=True`，
        而它在 Windows 上会**回退去读注册表**里的系统代理设置 ——
        该配置**不在环境变量里**（`env` 中没有任何 proxy 变量），所以光看环境变量发现不了。
        后果有二，都已实测：

        1. **数据路径多一跳**：embedding 请求携带的是**文档正文**，
           却要先交给那个代理进程；RAG 系统里这属于未记账的出站。
        2. **症状被改写**：服务没起时返回的是 **502**（代理替它答的）而不是
           `ConnectError` —— 会把人误导成"服务在跑但后端挂了"（本项目真的这么误判过一次）。

        实测对照（`127.0.0.1` 死端口）：`trust_env=True` → `HTTP 502`；
        `trust_env=False` → `ConnectError`。**loopback 就不该经过代理**，故这里不设开关。
        ⚠️ 若将来把 embedding 换到远程服务，本行需一并调整。
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=180.0,
                trust_env=not _is_loopback(self._base_url),
            )
        return self._client

    async def aclose(self) -> None:
        """关闭长生命周期客户端（应用 shutdown 调用；否则连接池泄漏）。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        client = self._get_client()
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result.extend(await self._embed_batch(client, batch))
        return result

    async def _embed_batch(self, client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
        resp = await client.post(
            f"{self._base_url}/api/embed",
            # keep_alive 随请求下发（而非只依赖 ollama serve 的环境变量）：
            # Ollama 默认空闲 5 分钟就卸载模型，之后第一个请求要重载（实测多等 ~2.5s）。
            json={"model": self._model, "input": texts, "keep_alive": self._keep_alive},
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [])
        if embeddings and len(embeddings[0]) != self._dim:
            raise ValueError(
                f"embedding 维度不匹配：模型返回 {len(embeddings[0])} 维，配置 EMBEDDING_DIM={self._dim}"
            )
        return embeddings
