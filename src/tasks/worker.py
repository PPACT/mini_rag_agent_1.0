"""Arq worker 入口（arq 0.28：自定义 WorkerSettings 类，用 `python -m arq` 启动）。"""
from __future__ import annotations

from arq.connections import RedisSettings

from src.config.settings import get_settings
from src.tasks import document_task


async def startup(ctx: dict) -> None:
    """worker 启动钩子（可预热连接池等）。"""


async def shutdown(ctx: dict) -> None:
    """worker 关闭钩子（清理连接）。"""


class WorkerSettings:
    """worker 配置。启动：`python -m arq src.tasks.worker.WorkerSettings`（项目根目录下）。"""

    functions = [document_task.process_document]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 3       # 失败重试次数，超过后 job 判失败
    job_timeout = 1800  # 单任务超时（秒），大文档向量化较慢
