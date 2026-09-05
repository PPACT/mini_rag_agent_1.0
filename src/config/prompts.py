"""提示词模板加载（外置 prompts/rag_system.yaml）。"""
from __future__ import annotations

from pathlib import Path

import yaml

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "rag_system.yaml"


def load_templates() -> tuple[str, str]:
    """返回 (system, human) 提示词模板。"""
    data = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))["rag_prompt"]
    return data["system"], data["human"]
