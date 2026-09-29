"""D9-⑧：embedding 必须是**单例**，且其长生命周期 HTTP 客户端要能被关闭。

原来 `get_embedding()` 每次调用都 new 一个实例，而该实例内部又
`async with httpx.AsyncClient()` —— 每个请求都新建客户端，**完全没有连接复用**。

⚠️ 与 reranker 的单例**差别**：本单例持有长生命周期 `AsyncClient`，
所以 shutdown 必须调 `close_embedding()`，否则是资源泄漏（reranker 持的是模型，无需关）。
"""
from __future__ import annotations

import asyncio

from src.embedding import base as emb
from src.embedding.ollama_embedding import OllamaEmbedding, _is_loopback


def test_get_embedding_is_singleton(monkeypatch):
    monkeypatch.setattr(emb, "_instance", None)
    assert emb.get_embedding() is emb.get_embedding(), "每请求新建实例 = 无连接复用"


def test_client_is_reused_across_calls():
    e = OllamaEmbedding()
    first = e._get_client()
    assert e._get_client() is first, "同一个实例内应复用同一个 AsyncClient"
    asyncio.run(e.aclose())


def test_aclose_releases_client_and_is_idempotent():
    e = OllamaEmbedding()
    e._get_client()
    asyncio.run(e.aclose())
    assert e._client is None
    asyncio.run(e.aclose())   # 再关一次不该炸


def test_client_recreated_after_close():
    """关掉之后还能继续用（重新建客户端），否则 shutdown 后的复用会炸。"""
    e = OllamaEmbedding()
    first = e._get_client()
    asyncio.run(e.aclose())
    second = e._get_client()
    assert second is not first and not second.is_closed
    asyncio.run(e.aclose())


def test_close_embedding_clears_singleton(monkeypatch):
    monkeypatch.setattr(emb, "_instance", None)
    inst = emb.get_embedding()
    inst._get_client()
    asyncio.run(emb.close_embedding())
    assert emb._instance is None
    assert inst._client is None, "close_embedding 必须真的关掉客户端，不能只丢引用"


def test_base_embedding_aclose_is_optional():
    """接口层面留了默认实现 —— 不持有资源的实现不必自己写。"""
    assert asyncio.run(emb.BaseEmbedding.aclose(object())) is None


# ===== 本机服务不走代理（D9 之外的新发现，见提交信息）=====
#
# httpx 默认 trust_env=True，且在 Windows 上会**回退读注册表**取系统代理 ——
# 即使 `env` 里一个 proxy 变量都没有（所以光看环境变量发现不了）。
# 于是 embedding 请求（携带**文档正文**）会先交给那个代理进程；
# 且服务没起时症状从 `ConnectError` 变成 **502**，足以误导排查。


def test_is_loopback_variants():
    for url in ("http://127.0.0.1:11451", "http://localhost:11451",
                "http://[::1]:11451", "http://0.0.0.0:11451"):
        assert _is_loopback(url), url
    for url in ("https://api.deepseek.com", "http://10.0.0.5:11434",
                "http://ollama.internal:11434"):
        assert not _is_loopback(url), url


def test_local_ollama_disables_env_proxy():
    """本机 embedding 必须绕开系统代理（loopback 不该经过任何代理）。"""
    e = OllamaEmbedding()
    assert e._get_client().trust_env is False
    asyncio.run(e.aclose())


def test_remote_embedding_still_honours_env_proxy(monkeypatch):
    """换成远程 embedding 服务时仍尊重环境代理 —— 别把隧道也一起关掉。"""
    e = OllamaEmbedding()
    monkeypatch.setattr(e, "_base_url", "http://ollama.internal:11434")
    assert e._get_client().trust_env is True
    asyncio.run(e.aclose())
