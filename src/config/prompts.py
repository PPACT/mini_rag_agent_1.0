"""提示词模板加载（外置 prompts/rag_system.yaml）。"""
from __future__ import annotations

from pathlib import Path

import yaml

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "rag_system.yaml"


def _load(section: str) -> dict:
    return yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))[section]


def load_templates() -> tuple[str, str]:
    """返回 RAG (system, human) 提示词模板。"""
    data = _load("rag_prompt")
    return data["system"], data["human"]


def load_rewrite_templates() -> tuple[str, str]:
    """返回多查询扩展 (system, human) 提示词模板。"""
    data = _load("query_rewrite")
    return data["system"], data["human"]


def load_rerank_templates() -> tuple[str, str]:
    """返回精排 (system, human) 提示词模板。"""
    data = _load("rerank")
    return data["system"], data["human"]


def load_ambiguity_check_templates() -> tuple[str, str]:
    """返回歧义判定 (system, human) 提示词模板。"""
    data = _load("ambiguity_check")
    return data["system"], data["human"]


def load_ambiguity_reply_templates() -> tuple[str, str]:
    """返回歧义澄清回复 (system, human) 提示词模板。"""
    data = _load("ambiguity_reply")
    return data["system"], data["human"]
