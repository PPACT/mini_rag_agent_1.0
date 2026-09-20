"""PostgreSQL 连接池（asyncpg）—— 按 kb 分别维护。

**两库分离**（真实业务库 / 压测库，详见 `docs/交流区.md` §1.6）：
每个 kb 一个独立连接池，连到各自的库。

⚠️ **`get_pool(kb)` 的 kb 必填、无默认值**：
一旦能默认（如 `kb=None → 真实库`），"忘传"就会**静默连到真实库**——
压测数据被灌进真实库且不报错。本项目反复踩的正是这类"静默走错分支"。
做成必填参数后，忘传在**类型/调用层面**就暴露；未知值直接抛 `UnknownKBError`。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from src.config.settings import get_settings
from src.db.kb import validate

_pools: dict[str, asyncpg.Pool] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _init_connection(conn: asyncpg.Connection) -> None:
    """每个新连接初始化检索参数。

    pgvector 的 hnsw.ef_search 默认 40，实测在 1231 切片时损失约 12% 的 ANN 召回；
    提到 100 仅多约 3ms 即可恢复满召回。在连接级设置，避免每次查询多一次往返。
    """
    ef = get_settings().hnsw_ef_search
    if ef:
        await conn.execute(f"SET hnsw.ef_search = {int(ef)}")


async def get_pool(kb: str) -> asyncpg.Pool:
    """返回指定 kb 的连接池（惰性初始化，按 kb 单例）。

    ⚠️ `kb` **必填**：不提供默认值。未知取值抛 `UnknownKBError`（fail-closed），
    绝不回退到任何库——**静默回退比报错危险得多**。
    """
    validate(kb)
    pool = _pools.get(kb)
    if pool is not None:
        return pool
    # 加锁：并发首次调用可能同时建池（原实现没有锁，会漏出一个池）
    lock = _locks.setdefault(kb, asyncio.Lock())
    async with lock:
        if kb not in _pools:
            _pools[kb] = await asyncpg.create_pool(
                dsn=get_settings().db_url(kb),
                min_size=1,
                max_size=10,
                init=_init_connection,
            )
        return _pools[kb]


async def close_pool(kb: str | None = None) -> None:
    """关闭连接池。

    - `kb=None`（默认）：关闭**全部**（应用 shutdown 用）
    - 指定 kb：只关那一个
    """
    if kb is None:
        for pool in list(_pools.values()):
            await pool.close()
        _pools.clear()
        return
    pool = _pools.pop(kb, None)
    if pool is not None:
        await pool.close()


@asynccontextmanager
async def transaction(kb: str) -> AsyncIterator[asyncpg.Connection]:
    """获取指定 kb 的连接并包裹在事务中（增删改一致性用）。"""
    pool = await get_pool(kb)
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
