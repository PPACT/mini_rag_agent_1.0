"""LangGraph Agent 编排（ReAct：LLM 按需调用 MCP 工具）。"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config.litellm_client import warn_if_thinking_enabled
from src.config.prompts import load_templates
from src.config.settings import get_settings

_agent = None


def _build_model() -> ChatOpenAI:
    """DeepSeek（OpenAI 兼容）作为 LangChain 聊天模型。后期切厂商只改这里/配置。

    ⚠️ D9-⑨：这里是**生成**路径 —— 超时/重试取自 `llm_timeout_generate` /
    `llm_max_retries_generate`（默认 20s × 1 次 → 最坏 ~40s）。
    原来是 `request_timeout=60` × `max_retries=3` = **最坏 180s+**，
    而这条路的正常耗时只有 ~2.5s：超时值相对正常耗时留了 24 倍，长尾全额转给用户。
    ⚠️ 与判定/精排路径（`litellm_client.complete`，8s × 0 次）**有意不同档**，不对齐。
    """
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        openai_api_key=s.llm_api_key,
        openai_api_base=s.llm_base_url,
        temperature=0.1,
        request_timeout=s.llm_timeout_generate,
        max_retries=s.llm_max_retries_generate,
    )


async def init_agent(tools: list) -> None:
    """用 MCP 工具 + 外置系统提示词构建 agent（应用 startup 时调用）。"""
    global _agent
    warn_if_thinking_enabled()   # D9-⑩：开启思考链时启动即告警
    if not get_settings().llm_api_key:
        # 未配置 key 时跳过构建（/chat 会在调用前先返回 503，入库链路不受影响）
        _agent = None
        return
    system, _ = load_templates()
    _agent = create_agent(_build_model(), tools, system_prompt=system)


def get_agent():
    if _agent is None:
        raise RuntimeError("Agent 未初始化")
    return _agent
