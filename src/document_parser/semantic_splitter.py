"""结构化切块：段落/句子边界 + 大小上限 + 重叠 + 标题/偏移元数据。"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """切块结果：文本 + 原文字符偏移 + 章节标题。"""

    text: str
    start: int
    end: int
    title: str | None = None


_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")


def _is_heading(line: str) -> bool:
    """粗判标题：短行且不以句末标点结尾。"""
    line = line.strip()
    if not line or len(line) > 40:
        return False
    return line[-1] not in "。！？；;.!?，,：:"


def _split_long(paragraph: str, base: int, chunk_size: int) -> list[tuple[str, int, int]]:
    """超长段落按句子切，返回 (text, start, end)，偏移相对原文。"""
    out: list[tuple[str, int, int]] = []
    cur, cur_start = "", base
    pos = base
    for s in _SENT_SPLIT.split(paragraph):
        if not s:
            continue
        s_start = pos
        pos += len(s)
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                out.append((cur, cur_start, cur_start + len(cur)))
            cur, cur_start = s, s_start
    if cur:
        out.append((cur, cur_start, cur_start + len(cur)))
    return out


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[TextChunk]:
    """结构优先切块：段落累积、检测标题、超长段按句子切，记录偏移与章节标题。"""
    text = text.rstrip()
    if not text.strip():
        return []

    # 拆段落 + 偏移
    paragraphs: list[tuple[str, int, int]] = []
    for m in re.finditer(r"[^\n]+", text):
        p = m.group().strip()
        if p:
            paragraphs.append((p, m.start(), m.end()))

    raw: list[tuple[str, int, int, str | None]] = []  # (text, start, end, title)
    cur_text, cur_start, cur_end, cur_title = "", 0, 0, None

    for p, s, e in paragraphs:
        if _is_heading(p):
            if cur_text:
                raw.append((cur_text, cur_start, cur_end, cur_title))
            cur_text, cur_start, cur_end, cur_title = p, s, e, p
            continue

        if cur_text and len(cur_text) + len(p) + 1 > chunk_size:
            raw.append((cur_text, cur_start, cur_end, cur_title))
            cur_text, cur_start, cur_end = "", s, e

        if len(p) > chunk_size:
            if cur_text:
                raw.append((cur_text, cur_start, cur_end, cur_title))
            for st, ss, se in _split_long(p, s, chunk_size):
                raw.append((st, ss, se, cur_title))
            cur_text, cur_start, cur_end = "", s, e
            continue

        if not cur_text:
            cur_start = s
        cur_text = (cur_text + "\n" + p).strip() if cur_text else p
        cur_end = e

    if cur_text:
        raw.append((cur_text, cur_start, cur_end, cur_title))

    # overlap：前一个 chunk 尾部拼到当前 chunk 头部（偏移仍指核心内容）
    if overlap > 0 and len(raw) > 1:
        out = [raw[0]]
        for i in range(1, len(raw)):
            tail = raw[i - 1][0][-overlap:]
            out.append((tail + "\n" + raw[i][0], raw[i][1], raw[i][2], raw[i][3]))
        raw = out

    return [TextChunk(text=t, start=s, end=e, title=ti) for t, s, e, ti in raw]
