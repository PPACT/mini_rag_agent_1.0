"""PostgreSQL 连接池（asyncpg）。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from src.config.settings import get_settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """每个新连接初始化检索参数。

    pgvector 的 hnsw.ef_search 默认 40，实测在 1231 切片时损失约 12% 的 ANN 召回；
    提到 100 仅多约 3ms 即可恢复满召回。在连接级设置，避免每次查询多一次往返。
    """
    ef = get_settings().hnsw_ef_search
    if ef:
        await conn.execute(f"SET hnsw.ef_search = {int(ef)}")


async def get_pool() -> asyncpg.Pool:
    """返回全局连接池（惰性初始化，单例）。"""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=get_settings().database_url,
            min_size=1,
            max_size=10,
            init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    """关闭连接池（应用 shutdown 时调用）。"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def transaction() -> AsyncIterator[asyncpg.Connection]:
    """获取连接并包裹在事务中（增删改一致性用）。"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
