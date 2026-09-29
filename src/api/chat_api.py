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
from src.observability.tracer import audit, audit_err
from src.rag.ambiguity import build_clarification, check_ambiguity
from src.rag.retriever import retrieve
from src.schemas.chat import ChatRequest, ChatResponse, ClarifyOption, Source

router = APIRouter(prefix="/chat", tags=["chat"])

# 生成环节返回空内容时的对外文案（P0-4）。
# ⚠️ 原来空答案会被**原样返回**：用户看到一片空白，日志里一条异常都没有 ——
#    同类静默降级，只是发生在生成侧而不是判定侧。
_NO_ANSWER_TEXT = (
    "抱歉，我**没能从检索到的资料里生成答案**（生成环节没有返回内容）。\n"
    "通常是资料里缺少能回答这个问题的内容 —— 换个问法、或补充更具体的线索可能有用。"
)

_UNVERIFIED_HEAD = (
    "⚠️ **下面这条答案没有得到确认**：一致性检查没能给出有效结论（{reason}），"
    "因此**无法排除**库中存在与之冲突的版本。请自行核对，或告诉我更多信息。"
)


def _annotate_unverified(answer: str, reason: str, chunks) -> str:
    """把「判定无法判断」**显式化**：答案照给，但标注 + 反问用户（P0-1）。

    为什么不是直接拒答：**判定器失败 ≠ 答案不存在**。直接拒答会把"护栏失灵"
    升级成"用户拿不到答案"（把可用性问题变成主要问题），而 P0-1 真正要消灭的是
    **静默** —— 这里让不确定性显式可见，同时保住主链路。

    ⚠️ 反问里只能给**检索命中的来源**，**不是**"互相矛盾的候选答案"：
    判定器 failure 时它没给出 `options`，所以这里**给不出真正的候选**。
    措辞上如实写成"检索到的来源"，不暗示做了一个没做的判断。
    """
    parts = [_UNVERIFIED_HEAD.format(reason=reason or "未返回有效结论"), "", answer]
    sources = sorted({c.source_file for c in chunks if c.source_file})
    if sources:
        parts += ["", "库中检索到这些来源，如果你要的是其中某一份，请指明："
                      + "；".join(sources[:5])]
    return "\n".join(parts)


def _cache_key(question: str, department: str, secret_level: int, top_k: int, kb: str) -> str:
    """缓存键。

    ⚠️ **必须含 kb**：两个库可能对同一问题给出完全不同的答案（真实库 vs 压测库），
    键里不带 kb 会让它们**共用缓存** —— 又是一次静默串库。

    ⚠️ D9-⑦：`kb` 还要进**键前缀**（`rag:{kb}:…`），不能只进哈希体 ——
    否则 `invalidate_cache` 无法按库扫描（原来只能 `rag:*` 全局清，
    压测库上传会冲掉真实库的缓存）。**前缀与失效范围必须同构**。
    """
    raw = json.dumps(
        {"q": question, "d": department, "s": secret_level, "k": top_k, "kb": kb},
        ensure_ascii=False,
    )
    return f"rag:{kb}:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


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

    # 3. 歧义判定（**三态**，P0-1）：明确判为"互相矛盾"才澄清；"无法判断"另走一路（见 4c）
    ambiguity = await check_ambiguity(req.question, chunks)
    if ambiguity.ambiguous:
        answer = await build_clarification(req.question, ambiguity)
        audit("clarify", question=req.question, user=user.name, options=len(ambiguity.options))
        resp = ChatResponse(
            answer=answer,
            sources=_to_sources(chunks),
            need_clarification=True,
            clarify_options=[ClarifyOption(source=o.source, summary=o.summary) for o in ambiguity.options],
            judge_status=ambiguity.status,
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

    # 4b/4c. 静默降级显式化（P0-1 / P0-4）—— 两者**互斥**：没有答案就谈不上"标注不确定"
    if not answer.strip():
        # 空答案不再原样返回给用户（原来是一片空白 + 日志无异常）
        answer = _NO_ANSWER_TEXT
        audit_err("answer_empty", question=req.question, user=user.name)
    elif ambiguity.unknown:
        # 判定器**没给出有效结论** → 答案照给，但标注 + 反问用户（不静默）
        answer = _annotate_unverified(answer, ambiguity.reason, chunks)
        audit_err("answer_unverified", question=req.question, user=user.name,
                  judge_reason=ambiguity.reason)

    # 5. 引用校验：sources 只返回 LLM **实际引用**的块（而非全部检索结果）
    cited = _cited_chunks(chunks, answer)
    audit("cite", question=req.question, cited=len(cited), total=len(chunks),
          cited_note="fallback_all" if len(cited) == len(chunks) and len(chunks) > 0 else "")

    # 6. 组装 + 写缓存
    resp = ChatResponse(answer=answer, sources=_to_sources(cited),
                        judge_status=ambiguity.status)
    await redis.set(key, resp.model_dump_json(), ex=settings.cache_ttl)
    return resp
