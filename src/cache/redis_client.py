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


async def invalidate_cache() -> None:
    """作废所有问答缓存（文档增删改后调用，避免旧结果命中）。"""
    redis = get_redis()
    keys = [k async for k in redis.scan_iter(match="rag:*")]
    if keys:
        await redis.delete(*keys)
