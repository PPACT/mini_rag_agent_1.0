"""LLM 统一收口（LiteLLM）。**换厂商只改配置，业务代码不动**。"""
from __future__ import annotations

import litellm

from src.config.settings import get_settings
from src.observability.tracer import audit


async def complete(messages: list[dict], temperature: float = 0.1) -> str:
    """调用 LLM 生成回答，返回文本内容。

    temperature 约定：**要求可复现的任务（精排、歧义判定）传 0**；
    需要多样性的任务（查询改写）保留默认 0.1；答案生成由 Agent 侧控制。
    """
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY 未配置，请先在 .env 填入")

    # 思考链：默认关闭。判定/精排都是分类任务，思考**既慢又会让 content 为空**
    # （思考与 content 共用 max_tokens 预算，吃满则 content 为空 → 调用方静默拿到空串）。
    extra: dict = {}
    if not settings.llm_thinking_enabled:
        # ⚠️ `reasoning_effort` 是 **DeepSeek 语义**。换 provider（如通用的 `openai/`）时，
        # 它可能被 `drop_params=True` **静默丢弃 → 思考链复活**。
        # 实测（logs/verify_provider_unbind.py）：deepseek/ 下 0 字思考 / 983ms；
        # 换 openai/ 后 7298 字思考 / 11591ms，且 finish=length。下面的告警就是兜这个底。
        extra["reasoning_effort"] = "none"

    resp = await litellm.acompletion(
        # provider 前缀决定用哪套**协议适配器**（可配，默认 "deepseek"——理由见 settings.py）
        model=f"{settings.llm_provider}/{settings.llm_model}",
        messages=messages,
        api_key=settings.llm_api_key,
        api_base=settings.llm_base_url,
        temperature=temperature,
        max_tokens=2048,
        timeout=60,  # 请求超时（秒）
        # 换用不支持该参数的模型时，让 litellm 丢掉它而不是直接报错
        drop_params=True,
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

    return msg.content or ""
