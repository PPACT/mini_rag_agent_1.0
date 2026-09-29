"""Redis 缓存层（问答结果缓存）。"""
from __future__ import annotations

import redis.asyncio as aioredis

from src.config.settings import get_settings

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """返回 Redis 客户端（惰性单例，decode_responses）。"""
    global _client
    if _client is None:
        _client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def invalidate_cache(kb: str) -> None:
    """作废**某个知识库**的问答缓存（该库文档增删改后调用）。

    ⚠️ D9-⑦：原来是 `match="rag:*"` —— **键含 kb、失效却是全局**，两者不对称。
    后果：**在压测库上传一次文档，真实库的问答缓存会被全部清掉**（只是白重算一次，
    不是正确性问题，但两个库会互相干扰，且压测流量会持续冲刷真实库的缓存）。

    现在按库失效。`kb` **必填**：给默认值就等于退回"静默全局失效"。
    """
    from src.db.kb import validate

    validate(kb)   # 拼错的 kb 会静默匹配不到任何键 → 这里直接报错
    redis = get_redis()
    keys = [k async for k in redis.scan_iter(match=f"rag:{kb}:*")]
    if keys:
        await redis.delete(*keys)
