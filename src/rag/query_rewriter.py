"""多查询扩展：用 LLM 把用户提问改写成多个检索友好的变体。

目的：用户提问口语化/含代称时，原始 query 的向量会偏离文档表述。
生成多个同义变体并行检索，可显著提升召回。
"""
from __future__ import annotations

import json
import re

from src.config.litellm_client import complete
from src.config.prompts import load_rewrite_templates
from src.observability.tracer import audit

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _parse_variants(raw: str, n: int) -> list[str]:
    """从 LLM 输出里解析 JSON 数组；解析失败返回空列表（由调用方回退）。"""
    match = _JSON_ARRAY.search(raw or "")
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()][:n]


async def expand_queries(question: str, n: int) -> list[str]:
    """返回 [原问题] + n 个改写变体（去重）。LLM 失败时安全回退为 [原问题]。"""
    if n <= 0:
        return [question]

    system_tpl, human_tpl = load_rewrite_templates()
    try:
        raw = await complete(
            [
                {"role": "system", "content": system_tpl.format(n=n)},
                {"role": "user", "content": human_tpl.format(question=question)},
            ]
        )
        variants = _parse_variants(raw, n)
        audit("query_rewrite", question=question, variants=len(variants))
    except Exception as e:  # noqa: BLE001
        # 改写是"锦上添花"，失败不能拖垮问答主链路
        audit("query_rewrite_failed", question=question, error=str(e))
        variants = []

    # 去重（保序），原问题恒在首位
    seen = {question}
    out = [question]
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
