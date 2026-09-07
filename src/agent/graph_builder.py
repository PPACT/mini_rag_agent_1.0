"""LangGraph Agent 编排（ReAct：LLM 按需调用 MCP 工具）。"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config.prompts import load_templates
from src.config.settings import get_settings

_agent = None


def _build_model() -> ChatOpenAI:
    """DeepSeek（OpenAI 兼容）作为 LangChain 聊天模型。后期切厂商只改这里/配置。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.deepseek_model,
        openai_api_key=s.deepseek_api_key,
        openai_api_base=s.deepseek_base_url,
        temperature=0.1,
        request_timeout=60,  # 请求超时（秒）
        max_retries=3,       # 失败自动重试
    )


async def init_agent(tools: list) -> None:
    """用 MCP 工具 + 外置系统提示词构建 agent（应用 startup 时调用）。"""
    global _agent
    if not get_settings().deepseek_api_key:
        # 未配置 key 时跳过构建（/chat 会在调用前先返回 503，入库链路不受影响）
        _agent = None
        return
    system, _ = load_templates()
    _agent = create_agent(_build_model(), tools, system_prompt=system)


def get_agent():
    if _agent is None:
        raise RuntimeError("Agent 未初始化")
    return _agent
