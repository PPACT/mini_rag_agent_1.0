"""FastAPI 入口。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.agent.graph_builder import init_agent
from src.agent.mcp_client_wrapper import close_mcp, get_tools, open_mcp
from src.api import chat_api, upload_api
from src.cache.redis_client import close_redis
from src.db.connection import close_pool
from src.tasks.queue import close_arq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：连接 MCP、构建 agent
    await open_mcp()
    await init_agent(await get_tools())
    yield
    # 关闭：清理资源
    await close_mcp()
    await close_arq()
    await close_redis()
    await close_pool()


app = FastAPI(title="RAG-Demo", lifespan=lifespan)
app.include_router(upload_api.router)
app.include_router(chat_api.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
