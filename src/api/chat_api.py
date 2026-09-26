"""RAG 问答接口。"""
from __future__ import annotations

import hashlib
import json
import re

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage

from src.agent.graph_builder import get_agent
from src.auth.deps import User, get_current_user
from src.cache.redis_client import get_redis
from src.config.prompts import load_templates
from src.config.settings import get_settings
from src.db.kb import KB_REAL
from src.observability.tracer import audit
from src.rag.ambiguity import build_clarification, check_ambiguity
from src.rag.retriever import retrieve
from src.schemas.chat import ChatRequest, ChatResponse, ClarifyOption, Source

router = APIRouter(prefix="/chat", tags=["chat"])


def _cache_key(question: str, department: str, secret_level: int, top_k: int, kb: str) -> str:
    """缓存键。

    ⚠️ **必须含 kb**：两个库可能对同一问题给出完全不同的答案（真实库 vs 压测库），
    键里不带 kb 会让它们**共用缓存** —— 又是一次静默串库。
    """
    raw = json.dumps(
        {"q": question, "d": department, "s": secret_level, "k": top_k, "kb": kb},
        ensure_ascii=False,
    )
    return "rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _to_sources(chunks) -> list[Source]:
    return [
        Source(source_file=c.source_file, chunk_index=c.chunk_index, score=round(c.score, 4))
        for c in chunks
    ]


# 只认明确的引用形式：[来源1] / [1] / 来源1 —— 必须带「来源」前缀或方括号，
# 否则答案正文里的裸数字（如"2 个工作日"）会被误当成引用编号。
_REF_RE = re.compile(r"\[来源\s*(\d+)\s*\]|\[(\d+)\]|来源\s*(\d+)", re.IGNORECASE)


def _cited_chunks(chunks, answer: str) -> list:
    """从答案里解析 [来源N] 引用，返回被实际引用的块（按上下文顺序）。

    命中规则：LLM 在 context 里看到的编号是「[来源1]..[来源N]」，对应 chunks 的 1-based 索引。
    若解析不到任何引用 → 保守回退为全部（避免前端空白），并记录审计。
    """
    cited = sorted({int(g) for m in _REF_RE.findall(answer) for g in m if g})
    idx = [i - 1 for i in cited if 1 <= i <= len(chunks)]
    if not idx:
        return chunks
    return [chunks[i] for i in idx]


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    """问答：缓存检查 → 检索 → 歧义判定 → (澄清 | Agent 生成) → 写缓存。

    department / secret_level 一律取自鉴权 token，不信任客户端参数。
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY 未配置，请先在 .env 填入")

    # 可见范围 = 公司级（全员）+ 本部门。缺少公司级会导致全员该看的制度被过滤掉。
    departments = list({user.department, settings.company_scope})
    secret_level = user.secret_level

    redis = get_redis()
    # /chat 是业务入口 → **固定查真实库**（压测库只在 /demo 与离线评测用）
    kb = KB_REAL
    key = _cache_key(req.question, user.department, user.secret_level, settings.top_k, kb)

    # 1. 缓存命中
    cached = await redis.get(key)
    if cached:
        audit("chat", hit_cache=True, question=req.question, user=user.name, department=user.department)
        return ChatResponse(**json.loads(cached))

    # 2. 检索
    context, chunks = await retrieve(req.question, departments, secret_level, kb=kb)
    audit("retrieve", question=req.question, user=user.name, department=user.department,
          kb=kb, hits=len(chunks))

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

    # 5. 引用校验：sources 只返回 LLM **实际引用**的块（而非全部检索结果）
    cited = _cited_chunks(chunks, answer)
    audit("cite", question=req.question, cited=len(cited), total=len(chunks),
          cited_note="fallback_all" if len(cited) == len(chunks) and len(chunks) > 0 else "")

    # 6. 组装 + 写缓存
    resp = ChatResponse(answer=answer, sources=_to_sources(cited))
    await redis.set(key, resp.model_dump_json(), ex=settings.cache_ttl)
    return resp
