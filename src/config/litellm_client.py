"""LLM 统一收口（LiteLLM）。**换厂商只改配置，业务代码不动**。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import litellm

from src.config.settings import get_settings
from src.observability.tracer import audit, audit_err

logger = logging.getLogger("rag.llm")

# 思考链告警只响一次（每进程）
_thinking_warned = False


def warn_if_thinking_enabled() -> None:
    """开启思考链时告警（**幂等**：每进程只响一次）。

    ⚠️ D9-⑩ 指出的缺口：现有 `thinking_leaked` 只在「**本该关、却泄漏了**」时响，
    **不覆盖「主动打开」的情况** —— 而主动打开恰恰是那个 bug 的原始触发条件：
    `LLM_THINKING_ENABLED=true` 而 `LLM_MAX_TOKENS` 还是 2048 时，
    思考会把预算吃满 → `finish_reason=length`、`content` 为空 →
    判定器 `parse_result("")` **静默判"不歧义"**（漏报）。

    由应用启动（`src/main.py`）与每次 `complete()` 调用两处触发，
    这样入口是 API 还是离线评测脚本都能被提醒到。
    """
    global _thinking_warned
    if _thinking_warned:
        return
    s = get_settings()
    if not s.llm_thinking_enabled:
        return
    _thinking_warned = True
    logger.warning(
        "LLM_THINKING_ENABLED=true —— 思考链已开启，已知后果："
        "① 判定/精排这类分类任务会慢数倍；"
        "② 思考与结论**共用 LLM_MAX_TOKENS=%s**，思考吃满则 content 为空 → "
        "判定器静默判「不歧义」→ **漏报**。若出现空结论，先调大 LLM_MAX_TOKENS。"
        "除对比评测外建议保持 false。",
        s.llm_max_tokens,
    )
    audit("thinking_enabled_warning", provider=s.llm_provider,
          model=s.llm_model, max_tokens=s.llm_max_tokens)


@dataclass(frozen=True)
class LLMReply:
    """一次 LLM 调用的结果 **+ 元信息**（P0-1 / P0-4）。

    存在的唯一理由：**「模型没答」必须与「模型答了」可区分**。

    ⚠️ 原来 `complete()` 结尾是 `return msg.content or ""` —— **`finish_reason` 被丢弃**，
    于是「被 `max_tokens` 截断」与「模型本来就没答」在代码里长得一模一样，
    调用方只能拿到一个空串，然后**静默继续**。
    而 P0-1 明确要求把 `finish_reason=length` 判为"无法判断" ——
    **不把元信息带出来，那条要求根本实现不了。**
    """

    text: str
    finish_reason: str | None = None

    @property
    def answered(self) -> bool:
        """模型是否给出了**可用内容**（纯空白不算答了）。"""
        return bool(self.text.strip())

    @property
    def truncated(self) -> bool:
        """是否因 `max_tokens` 被截断（这类"没答"与"模型确实无话可说"不是一回事）。"""
        return (self.finish_reason or "").lower() == "length"

    @property
    def empty_reason(self) -> str | None:
        """没答的原因（答了则为 None）。"""
        if self.answered:
            return None
        return "truncated" if self.truncated else "empty"


async def complete_with_meta(messages: list[dict], temperature: float = 0.1) -> LLMReply:
    """调用 LLM，返回**内容 + 元信息**。调用方**必须**判 `answered` 再决定怎么用。

    需要区分"没答"与"答了"的调用点用这个；只想要文本、且拿空串也无所谓的用 `complete()`。
    """
    """调用 LLM 生成回答，返回文本内容。

    temperature 约定：**要求可复现的任务（精排、歧义判定）传 0**；
    需要多样性的任务（查询改写）保留默认 0.1；答案生成由 Agent 侧控制。

    ⚠️ 本函数是**延迟敏感**路径（判定 / 精排）的收口：超时与重试取自
    `llm_timeout_judge` / `llm_max_retries_judge`（D9-⑨），**默认 8s 且不重试** ——
    生成路径（Agent）走 `graph_builder`，另有更宽的一档，两者**不做对齐**。
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY 未配置，请先在 .env 填入")
    warn_if_thinking_enabled()

    # 思考链：默认关闭。判定/精排都是分类任务，思考**既慢又会让 content 为空**
    # （思考与 content 共用 max_tokens 预算，吃满则 content 为空 → 调用方静默拿到空串）。
    extra: dict = {}
    if not settings.llm_thinking_enabled:
        # ⚠️ `reasoning_effort` 是 **DeepSeek 语义**。换 provider（如通用的 `openai/`）时，
        # 它可能被 `drop_params=True` **静默丢弃 → 思考链复活**。
        # 实测（eval/verify_provider_unbind.py）：deepseek/ 下 0 字思考 / 983ms；
        # 换 openai/ 后 7298 字思考 / 11591ms，且 finish=length。下面的告警就是兜这个底。
        extra["reasoning_effort"] = "none"

    resp = await litellm.acompletion(
        # provider 前缀决定用哪套**协议适配器**（可配，默认 "deepseek"——理由见 settings.py）
        model=f"{settings.llm_provider}/{settings.llm_model}",
        messages=messages,
        api_key=settings.llm_api_key,
        api_base=settings.llm_base_url,
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,   # D9-⑩：原为硬编码 2048
        timeout=settings.llm_timeout_judge,   # D9-⑨：原为硬编码 60（等于没有超时）
        num_retries=settings.llm_max_retries_judge,  # 显式 0，不再依赖 litellm 默认值
        # ⚠️ **默认 False = 不静默丢弃**（`2.0-41`）。
        # 原来是写死的 `True`：换 provider 后 `reasoning_effort="none"` 会被**静默丢掉**
        # → **思考链复活且不报错**（D8 踩过的那个坑）。
        # 现在不支持就**报错** —— 从"静默失效"变成"响"。
        # ⚠️ 换 provider 后若这里报错，**那是设计如此**：去 `2.0-42` 补翻译，别改回 True。
        drop_params=settings.llm_drop_params,
        **extra,
    )
    msg = resp.choices[0].message

    # ⚠️ 兜底告警：本该关思考，却收到了思考内容 → 说明 provider 适配器把
    # `reasoning_effort="none"` 丢了（**换厂商时最容易踩的坑**）。
    # 必须让它"响"：否则表现为"莫名慢 10 秒 + 偶尔结论为空"，几周后才被发现。
    leaked = getattr(msg, "reasoning_content", None) or ""
    if not settings.llm_thinking_enabled and leaked:
        audit("thinking_leaked", provider=settings.llm_provider,
              model=settings.llm_model, reasoning_chars=len(leaked))

    return LLMReply(text=msg.content or "", finish_reason=getattr(resp.choices[0], "finish_reason", None))


async def complete(messages: list[dict], temperature: float = 0.1) -> str:
    """`complete_with_meta()` 的**文本兼容版**（原签名，5 个 eval 脚本仍在使用）。

    ⚠️ **中心化静默降级告警**：本函数会把"没答"记为 **ERR 审计**。
    这样即使某个调用点没迁去用 `complete_with_meta()`、仍然只拿一个空串，
    「模型没答」这件事也**一定会在日志里留下痕迹**（P0-4 要堵的就是这个洞）。
    与调用点自己的专项审计（如 `rerank_empty` / `ambiguity_unparsed`）**是互补的**：
    这里是"任何路径都不静默"的兜底，那里带具体上下文。
    """
    reply = await complete_with_meta(messages, temperature=temperature)
    if not reply.answered:
        audit_err("llm_empty_reply", reason=reply.empty_reason,
                  finish_reason=reply.finish_reason)
    return reply.text
