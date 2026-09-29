"""结构化切块：段落/句子边界 + 大小上限 + 比例重叠 + 标题/偏移元数据。"""
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

# 调参常量
_FLUSH_RATIO = 0.5     # 遇到标题时，当前块达到 chunk_size 的该比例才另起一块
_TINY_RATIO = 0.15     # 小于 chunk_size 该比例的块视为过小，尝试向后合并
_OVERLAP_CAP = 0.25    # 重叠最多取前一块长度的该比例，避免短块重复过多


def _is_heading(line: str) -> bool:
    """粗判标题：短行且不以句末标点结尾。"""
    line = line.strip()
    if not line or len(line) > 40:
        return False
    return line[-1] not in "。！？；;.!?，,：:"


def _split_long(paragraph: str, base: int, chunk_size: int) -> list[list]:
    """超长段落按句子切，返回 [text, start, end, title] 列表，偏移相对原文。"""
    out: list[list] = []
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
                out.append([cur, cur_start, cur_start + len(cur), None])
            cur, cur_start = s, s_start
    if cur:
        out.append([cur, cur_start, cur_start + len(cur), None])
    return out


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[TextChunk]:
    """结构优先切块。

    规则：
    1. 段落累积，标题作为章节标记；
    2. 标题只在当前块"已足够大"时才另起新块，避免标题单独成废块；
    3. 超长段落按句子切；
    4. 过小的块向后合并；
    5. 重叠按前一块长度的比例封顶，避免短块大比例重复。
    """
    text = text.rstrip()
    if not text.strip():
        return []

    blocks = [
        (m.group().strip(), m.start(), m.end(), _is_heading(m.group().strip()))
        for m in re.finditer(r"[^\n]+", text)
        if m.group().strip()
    ]

    chunks: list[list] = []  # [text, start, end, title]
    cur: list | None = None

    def flush() -> None:
        nonlocal cur
        if cur and cur[0].strip():
            chunks.append(cur)
        cur = None

    for p, s, e, is_head in blocks:
        if is_head:
            # 只有当前块够大才另起新块；否则继续累积（标题仅更新章节标记）
            if cur is not None and len(cur[0]) >= chunk_size * _FLUSH_RATIO:
                flush()
            if cur is None:
                cur = [p, s, e, p]
            else:
                cur[0] = f"{cur[0]}\n{p}"
                cur[2] = e
                if not cur[3]:
                    cur[3] = p
            continue

        # 超长段落先按句子切（无论当前是否已有累积块）
        if len(p) > chunk_size:
            title = cur[3] if cur is not None else None
            flush()
            for part in _split_long(p, s, chunk_size):
                part[3] = title
                chunks.append(part)
            continue

        if cur is None:
            cur = [p, s, e, None]
            continue

        if len(cur[0]) + len(p) + 1 > chunk_size:
            flush()
            cur = [p, s, e, None]
        else:
            cur[0] = f"{cur[0]}\n{p}"
            cur[2] = e

    flush()

    # 过小的块向后合并（含首个块过小的情况）
    chunks = _merge_tiny(chunks, chunk_size)

    # 重叠：按前一块长度比例封顶
    if overlap > 0 and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1][0]
            eff = min(overlap, max(0, int(len(prev) * _OVERLAP_CAP)))
            if eff:
                with_overlap.append(
                    [f"{prev[-eff:]}\n{chunks[i][0]}", chunks[i][1], chunks[i][2], chunks[i][3]]
                )
            else:
                with_overlap.append(chunks[i])
        chunks = with_overlap

    return [TextChunk(text=t, start=s, end=e, title=ti) for t, s, e, ti in chunks]


def _merge_tiny(chunks: list[list], chunk_size: int) -> list[list]:
    """把过小的块合并到相邻块（优先向后并入，避免产生废块）。"""
    if not chunks:
        return []
    threshold = chunk_size * _TINY_RATIO
    out: list[list] = []
    for c in chunks:
        if out and len(c[0]) < threshold and len(out[-1][0]) + len(c[0]) + 1 <= chunk_size:
            out[-1][0] = f"{out[-1][0]}\n{c[0]}"
            out[-1][2] = c[2]
        else:
            out.append(c)
    # 若首块仍过小且后面还有块，把它并入后一块
    if len(out) > 1 and len(out[0][0]) < threshold and len(out[0][0]) + len(out[1][0]) + 1 <= chunk_size:
        out[1][0] = f"{out[0][0]}\n{out[1][0]}"
        out[1][1] = out[0][1]
        if not out[1][3]:
            out[1][3] = out[0][3]
        out.pop(0)
    return out
