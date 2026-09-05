"""LLM 统一收口（LiteLLM）。后期切换云端模型只改配置，业务代码不动。"""
from __future__ import annotations

import litellm

from src.config.settings import get_settings


async def complete(messages: list[dict], temperature: float = 0.1) -> str:
    """调用 LLM 生成回答，返回文本内容。"""
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，请先在 .env 填入")

    resp = await litellm.acompletion(
        model=f"deepseek/{settings.deepseek_model}",
        messages=messages,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""
