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


async def enqueue_process_document(
    kb: str, document_id: str, department: str | None = None, secret_level: int = 0
) -> None:
    """入队文档处理任务（附带知识库与权限元数据）。

    ⚠️ `kb` **必填**（真实 / 压测）：worker 按它决定往哪个库写。
    不设默认值——否则压测数据可能被**静默灌进真实库**。

    ⚠️ **一律用关键字传参**（D9-①）：位置传参下，只要有人在 worker 签名中间插一个
    参数，`kb` 就会**静默错位 → 静默灌错库**，且全程不报错。kwargs 下 arq 按名绑定，
    插入新参数不再影响语义。
    代价：**改名**这类漂移失去了位置对齐的隐式保护 —— 由 `tests/test_queue.py`
    的「入队键集合 == worker 形参名集合」契约测试兜住。
    """
    arq = await get_arq()
    await arq.enqueue_job(
        "process_document",
        kb=kb,
        document_id=document_id,
        department=department,
        secret_level=secret_level,
    )
