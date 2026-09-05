"""结构化切块：按段落/句子边界 + 大小上限 + 重叠（语义切块的轻量实现）。"""
from __future__ import annotations

import re


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """把文本切成若干 chunk：优先按段落边界，超长段落按句子切，末尾拼接 overlap。"""
    text = text.strip()
    if not text:
        return []

    # 1. 按段落
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]

    # 2. 段落累积成 chunk（不超过 chunk_size）
    raw: list[str] = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = f"{cur}\n{p}".strip() if cur else p
        else:
            if cur:
                raw.append(cur)
                cur = ""
            if len(p) <= chunk_size:
                cur = p
            else:
                raw.extend(_split_long(p, chunk_size))
    if cur:
        raw.append(cur)

    # 3. overlap：把上一个 chunk 的尾部拼到当前 chunk 头部（保持上下文连续）
    if overlap <= 0 or len(raw) <= 1:
        return raw
    result = [raw[0]]
    for i in range(1, len(raw)):
        tail = raw[i - 1][-overlap:]
        result.append(f"{tail}\n{raw[i]}")
    return result


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    """超长段落按句子边界切分。"""
    sentences = re.split(r"(?<=[。！？!?；;])", paragraph)
    out: list[str] = []
    cur = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out
