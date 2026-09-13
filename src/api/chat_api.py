"""RAG 问答接口。"""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage

from src.agent.graph_builder import get_agent
from src.auth.deps import User, get_current_user
from src.cache.redis_client import get_redis
from src.config.prompts import load_templates
from src.config.settings import get_settings
from src.observability.tracer import audit
from src.rag.ambiguity import build_clarification, check_ambiguity
from src.rag.retriever import retrieve
from src.schemas.chat import ChatRequest, ChatResponse, ClarifyOption, Source

router = APIRouter(prefix="/chat", tags=["chat"])


def _cache_key(question: str, department: str, secret_level: int, top_k: int) -> str:
    raw = json.dumps({"q": question, "d": department, "s": secret_level, "k": top_k}, ensure_ascii=False)
    return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _to_sources(chunks) -> list[Source]:
    return [
        Source(source_file=c.source_file, chunk_index=c.chunk_index, score=round(c.score, 4))
        for c in chunks
    ]


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    """问答：缓存检查 → 检索 → 歧义判定 → (澄清 | Agent 生成) → 写缓存。

    department / secret_level 一律取自鉴权 token，不信任客户端参数。
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置，请先在 .env 填入")

    departments = [user.department]
    secret_level = user.secret_level

    redis = get_redis()
    key = _cache_key(req.question, user.department, user.secret_level, settings.top_k)

    # 1. 缓存命中
    cached = await redis.get(key)
    if cached:
        audit("chat", hit_cache=True, question=req.question, user=user.name, department=user.department)
        return ChatResponse(**json.loads(cached))

    # 2. 检索
    context, chunks = await retrieve(req.question, departments, secret_level)
    audit("retrieve", question=req.question, user=user.name, department=user.department, hits=len(chunks))

    # 3. 歧义判定：候选互相矛盾时不擅自选一个，改为温和澄清
    ambiguity = await check_ambiguity(req.question, chunks)
    if ambiguity.ambiguous:
        answer = await build_clarification(req.question, ambiguity)
        audit("clarify", question=req.question, user=user.name, options=len(ambiguity.options))
        resp = ChatResponse(
            answer=answer,
            sources=_to_sources(chunks),
            need_clarification=True,
            clarify_options=[ClarifyOption(source=o.source, summary=o.summary) for o in ambiguity.options],
        )
        await redis.set(key, resp.model_dump_json(), ex=settings.cache_ttl)
        return resp

    # 4. 正常生成（系统提示词在 graph_builder 注入，这里只传上下文+问题）
    _, human_tpl = load_templates()
    human_msg = HumanMessage(content=human_tpl.format(context=context, question=req.question))
    agent = get_agent()
    result = await agent.ainvoke({"messages": [human_msg]})
    answer = str(result["messages"][-1].content)
    audit("llm", question=req.question, user=user.name, answer_len=len(answer))

    # 5. 组装 + 写缓存
    resp = ChatResponse(answer=answer, sources=_to_sources(chunks))
    await redis.set(key, resp.model_dump_json(), ex=settings.cache_ttl)
    return resp
