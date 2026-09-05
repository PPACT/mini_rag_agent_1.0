"""RAG 问答接口。"""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from src.agent.graph_builder import get_agent
from src.cache.redis_client import get_redis
from src.config.prompts import load_templates
from src.config.settings import get_settings
from src.observability.tracer import audit
from src.rag.retriever import retrieve
from src.schemas.chat import ChatRequest, ChatResponse, Source

router = APIRouter(prefix="/chat", tags=["chat"])


def _cache_key(question: str, departments, secret_level, top_k: int) -> str:
    raw = json.dumps({"q": question, "d": departments, "s": secret_level, "k": top_k}, ensure_ascii=False)
    return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """问答：缓存检查 -> 检索 -> Agent（按需调 MCP 工具）-> 生成 -> 写缓存。"""
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置，请先在 .env 填入")

    redis = get_redis()
    key = _cache_key(req.question, req.departments, req.secret_level, settings.top_k)

    # 1. 缓存命中
    cached = await redis.get(key)
    if cached:
        audit("chat", hit_cache=True, question=req.question)
        return ChatResponse(**json.loads(cached))

    # 2. 检索
    context, chunks = await retrieve(req.question, req.departments, req.secret_level)
    audit("retrieve", question=req.question, hits=len(chunks))

    # 3. Agent 生成（系统提示词在 graph_builder 注入，这里只传上下文+问题）
    _, human_tpl = load_templates()
    human_msg = HumanMessage(content=human_tpl.format(context=context, question=req.question))
    agent = get_agent()
    result = await agent.ainvoke({"messages": [human_msg]})
    answer = str(result["messages"][-1].content)
    audit("llm", question=req.question, answer_len=len(answer))

    # 4. 组装 + 写缓存
    sources = [
        Source(source_file=c.source_file, chunk_index=c.chunk_index, score=round(c.score, 4))
        for c in chunks
    ]
    resp = ChatResponse(answer=answer, sources=sources)
    await redis.set(key, resp.model_dump_json(), ex=settings.cache_ttl)
    return resp
