"""Arq 任务队列（入队侧）。"""
from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from src.config.settings import get_settings

_arq: ArqRedis | None = None


async def get_arq() -> ArqRedis:
    global _arq
    if _arq is None:
        _arq = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _arq


async def close_arq() -> None:
    global _arq
    if _arq is not None:
        await _arq.aclose()
        _arq = None


async def enqueue_process_document(document_id: str, department: str | None = None, secret_level: int = 0) -> None:
    """入队文档处理任务（附带权限元数据）。"""
    arq = await get_arq()
    await arq.enqueue_job("process_document", document_id, department, secret_level)
