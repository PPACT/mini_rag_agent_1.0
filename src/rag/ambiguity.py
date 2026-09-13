"""歧义判定与澄清：避免"候选互相矛盾却擅自选一个"。

背景：当库中存在大量语义等价文档（如各部门的同一制度各有微差版本）时，
检索各环节指标都好看，但用户可能拿到**别人的版本**——指标全绿、业务全错。
此时正确行为是**温和澄清**，而不是硬答。

设计：
- **门控**（便宜）：top-K 来源不够分散时直接跳过，省一次 LLM 调用
- **判定**：LLM 只判断"候选之间是否互相矛盾"（纯分类任务，易结构化）
- **回退**：任何异常/解析失败 → 判定为"无歧义"（宁可漏报，不可误伤）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.config.litellm_client import complete
from src.config.prompts import load_ambiguity_check_templates, load_ambiguity_reply_templates
from src.config.settings import get_settings
from src.observability.tracer import audit
from src.vector_store.base import Chunk

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ClarifyOption:
    """一个候选答案（用于向用户展示、让其明确意图）。"""

    source: str
    summary: str


@dataclass
class AmbiguityResult:
    ambiguous: bool = False
    reason: str = ""
    options: list[ClarifyOption] = field(default_factory=list)


def is_diverse(chunks: list[Chunk], threshold: int) -> bool:
    """门控信号：top-K 是否来自 >= threshold 个不同来源。"""
    if not chunks:
        return False
    return len({c.source_file for c in chunks if c.source_file}) >= threshold


def _format_candidates(chunks: list[Chunk], snippet: int) -> str:
    return "\n\n".join(
        f"[{i}] (来源: {c.source_file})\n{c.content[:snippet]}"
        for i, c in enumerate(chunks, 1)
    )


def parse_result(raw: str) -> AmbiguityResult:
    """解析 LLM 输出；无法解析时返回"无歧义"（安全回退）。"""
    match = _JSON_OBJ.search(raw or "")
    if not match:
        return AmbiguityResult()
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return AmbiguityResult()
    if not isinstance(data, dict) or not data.get("ambiguous"):
        return AmbiguityResult()

    options: list[ClarifyOption] = []
    for opt in data.get("options") or []:
        if isinstance(opt, dict) and opt.get("summary"):
            options.append(
                ClarifyOption(source=str(opt.get("source", "")), summary=str(opt["summary"]))
            )
    return AmbiguityResult(ambiguous=True, reason=str(data.get("reason", "")), options=options)


async def check_ambiguity(question: str, chunks: list[Chunk]) -> AmbiguityResult:
    """判定候选之间是否存在互相矛盾的答案。失败/不确定时返回"无歧义"。"""
    settings = get_settings()
    if not settings.ambiguity_check_enabled or len(chunks) < 2:
        return AmbiguityResult()
    # 门控：来源不够分散 → 跳过 LLM 判定（绝大多数普通查询在此被省掉）
    if not is_diverse(chunks, settings.ambiguity_source_threshold):
        return AmbiguityResult()

    system_tpl, human_tpl = load_ambiguity_check_templates()
    try:
        raw = await complete(
            [
                {"role": "system", "content": system_tpl},
                {
                    "role": "user",
                    "content": human_tpl.format(
                        question=question,
                        candidates=_format_candidates(chunks, settings.ambiguity_snippet_chars),
                    ),
                },
            ]
        )
        result = parse_result(raw)
        audit("ambiguity_check", question=question, ambiguous=result.ambiguous, options=len(result.options))
        return result
    except Exception as e:  # noqa: BLE001
        audit("ambiguity_check_failed", question=question, error=str(e))
        return AmbiguityResult()


async def build_clarification(question: str, result: AmbiguityResult) -> str:
    """生成温和的澄清回复。失败时回退到模板化文案（不阻断主链路）。"""
    system_tpl, human_tpl = load_ambiguity_reply_templates()
    options_text = "\n".join(f"- {o.source}：{o.summary}" for o in result.options)
    try:
        return await complete(
            [
                {"role": "system", "content": system_tpl},
                {
                    "role": "user",
                    "content": human_tpl.format(question=question, reason=result.reason, options=options_text),
                },
            ]
        )
    except Exception as e:  # noqa: BLE001
        audit("ambiguity_reply_failed", question=question, error=str(e))
        return "我找到了多份相关文档，它们给出的答案不一致：\n" + options_text + "\n\n请问你指的是哪一种？"
