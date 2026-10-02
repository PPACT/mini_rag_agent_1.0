r"""从纯文本里**重建** Markdown 表格 —— 补 `2.0-1` 的分母缺口。

（本 docstring 开头带 **r 前缀**（raw string）：正文里写了 `\|` 这类转义序列，
不带 raw 的话 Python 会报 DeprecationWarning: invalid escape sequence
—— 正是 `tests/test_syntax_sweep.py` 抓出来的。
⚠️ 顺带一条**写文档时的坑**：这段解释本身**不能在正文里写出三引号**，
否则它会**提前结束 docstring** —— 我第一版就是这么把它弄成语法错误的。）

## 为什么需要它

md→docx 的转换工具把**表格降级成了普通段落**（`| a | b |`）。于是在解析层看来
那些文件"没有表格"（Word 的 `<w:tbl>` = 0、本批 PDF 无框线 → `extract_tables()` 0 张）。
**但表格其实在文本里，而且格式还是标准 Markdown（带分隔行）。**

实测（`docs/local/corpus/`）：

```
07_2025年Q3财务简报.docx      22 条管道行  列数分布 [(4, 22)]  分隔行 3
09_研发部季度汇报_2025Q3.docx  31 条        3/4/6 混杂        分隔行 4
01_差旅与报销管理制度.pdf       0 条          ← PDF 里连 | 都没有，**救不了**
```

⚠️ **重建只救得了 Word/txt/md，救不了 PDF** —— 报告里的命中率数字
**必须写清"分母只覆盖 Word/Excel/txt，不含 PDF"**，否则又是一个"看似可信、其实不完整"的数。

## ⭐ 判据分两档（交流区 §1.34⑤-d 的裁决）—— **不许混用**

| 档位 | 判据 | 取向 |
|---|---|---|
| **重建**（真的转成 `table` 块） | 连续 ≥2 行 + 每行 ≥2 个 `\|` + **列数一致** + **有分隔行** | **宁可漏判** —— 误判 = **静默污染知识库** |
| **flag**（只记一行"疑似表格"） | 含 3+ 个 `\|` 的行 ≥ 2 行 | **宁可多记** —— 不改变任何行为 |

混用的后果：**记的时候太少、改的时候太莽**。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.document_parser.blocks import Block, ParsedDoc
from src.document_parser.tables import is_complex, render_nl, rows_to_markdown

# 分隔行：`| --- | --- |` / `|:--|--:|` —— 只有 `|`、`-`、`:`、空白
_SEP_ROW = re.compile(r"[\s|:\-]+")


def _is_pipe_row(line: str) -> bool:
    """管道行：至少 2 个 `|`（1 个的通常是正文里用了个竖线，不是表）。"""
    return line.count("|") >= 2


def _is_sep_row(line: str) -> bool:
    s = line.strip()
    return bool(s) and "|" in s and bool(_SEP_ROW.fullmatch(s))


def _cells(line: str) -> list[str]:
    """把一行拆成单元格（去掉首尾的 `|`）。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


@dataclass
class RecoveredTable:
    """一段可重建的表格（行已去掉分隔行，表头在前）。"""

    rows: list[list[str]]
    line_start: int          # 在块内文本里的行号（含）
    line_end: int            # 行号（不含）


@dataclass
class SuspectRange:
    """**疑似**表格但**不可重建**的区间 —— 只记 flag，不改文本。"""

    line_start: int
    line_end: int
    reason: str


@dataclass
class RecoveryReport:
    recovered: int = 0
    suspect: list[SuspectRange] = field(default_factory=list)
    blocks_before: int = 0
    blocks_after: int = 0


def find_tables(lines: list[str]) -> list[RecoveredTable]:
    """按**严判据**找出可重建的表格。

    一个连续管道行段里**允许含多张表**（`07` 的 22 行里就有 3 张）——
    切分依据是**分隔行**：分隔行**前一行**是那张表的表头，分隔行**之后**是它的数据。
    """
    out: list[RecoveredTable] = []
    i, n = 0, len(lines)
    while i < n:
        if not _is_pipe_row(lines[i]):
            i += 1
            continue
        j = i
        while j < n and _is_pipe_row(lines[j]):
            j += 1
        out.extend(_tables_in_run(lines, i, j))
        i = j
    return out


def _tables_in_run(lines: list[str], start: int, end: int) -> list[RecoveredTable]:
    run = lines[start:end]
    if len(run) < 2:                                   # 判据：连续 ≥2 行
        return []
    rows = [_cells(l) for l in run]
    if len({len(r) for r in rows}) != 1:               # 判据：列数一致
        return []
    if len(rows[0]) < 2:                               # 判据：每行 ≥2 个 |
        return []
    seps = [k for k, l in enumerate(run) if _is_sep_row(l)]
    if not seps:                                       # 判据：必须有分隔行
        return []

    out: list[RecoveredTable] = []
    cursor = 0                                          # 上一张表的数据结束位置
    for si, sep in enumerate(seps):
        header = rows[cursor:sep]                       # 分隔行之前 = 表头（可多行）
        data_end = (seps[si + 1] - 1) if si + 1 < len(seps) else len(rows)
        data = [r for k, r in enumerate(rows[sep + 1:data_end], sep + 1)
                if not _is_sep_row(run[k])]
        if header and data:
            out.append(RecoveredTable(rows=header + data,
                                      line_start=start + cursor,
                                      line_end=start + data_end))
        cursor = data_end
    return out


def find_suspects(lines: list[str]) -> list[SuspectRange]:
    """按**宽判据**找出"疑似表格但不可重建"的区间（只记，不改）。

    宽判据 = 含 3+ 个 `|` 的行 ≥ 2 行。**故意比重建判据宽** ——
    它的代价只是多记一行，而它的价值是**让"没被识别的表格"可见**（不静默）。
    """
    out: list[SuspectRange] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].count("|") < 3:
            i += 1
            continue
        j = i
        while j < n and lines[j].count("|") >= 3:
            j += 1
        if j - i >= 2:
            run = lines[i:j]
            rows = [_cells(l) for l in run]
            # ⚠️ **列出所有**不满足的条件，不只报第一个 ——
            # 否则人看到"缺分隔行"、补上之后仍然建不出来（因为列数也不齐），白跑一轮。
            reasons: list[str] = []
            if not any(_is_sep_row(l) for l in run):
                reasons.append("缺分隔行（|---| 边界）")
            if len({len(r) for r in rows}) != 1:
                reasons.append(f"列数不一致（{sorted({len(r) for r in rows})}）")
            if len(rows[0]) < 2:
                reasons.append("列太少（<2）")
            out.append(SuspectRange(line_start=i, line_end=j,
                                    reason="；".join(reasons) or "形状不完整"))
        i = j
    return out


def recover_tables(parsed: ParsedDoc) -> tuple[ParsedDoc, RecoveryReport]:
    """把 `paragraph` 块里可重建的 Markdown 表格**转成 `table` 块**。

    ⚠️ **两个容易写错的地方**（都是实测撞出来的，不是想当然）：

    1. **表格可能横跨多个块**。Word 里**每一行是一个独立的 `paragraph` 块**
       （实测 `07`：`#7` 表头 / `#8` 分隔行 / `#9..#13` 数据 → 一张表占 **7 个块**）。
       所以**不能只在单个块内找** —— 那样 Word 的表**一张都重建不出来**（我第一版就是这么错的）。
    2. **表格也可能整个装在一个块里**（txt/md：一张表就是一个块）。
       → 所以实现要**按行流处理**，同时**保留原始块边界**，两种形态才都正确。

    ⚠️ **只处理 `paragraph` 块**：
    - `table` 块已经是表（不重复处理）；
    - `code` 块里的 `|` 是代码（SQL 的 `||`、位运算 `|`）—— **碰它必出事**。
    - 非 paragraph 块同时是**屏障**：表格**不许跨过**它们（否则可能把两段无关文本缝成一张表）。

    不可重建但疑似表格的区间**只记 warning**（宽判据），文本一字不动。
    """
    report = RecoveryReport(blocks_before=len(parsed.blocks))
    new_blocks: list[Block] = []
    i, n = 0, len(parsed.blocks)

    while i < n:
        if parsed.blocks[i].kind != "paragraph":       # 非文本块：原样通过 + 作屏障
            new_blocks.append(parsed.blocks[i])
            i += 1
            continue
        j = i
        while j < n and parsed.blocks[j].kind == "paragraph":
            j += 1
        new_blocks.extend(_process_run(parsed.blocks[i:j], parsed, report))
        i = j

    report.blocks_after = len(new_blocks)
    return ParsedDoc(blocks=new_blocks, parser=parsed.parser, pages=parsed.pages,
                     warnings=parsed.warnings), report


def _process_run(run: list[Block], parsed: ParsedDoc, report: RecoveryReport) -> list[Block]:
    """处理一段**连续的** `paragraph` 块：展平成行流找表，再按原块边界重建。"""
    lines: list[str] = []
    owner: list[int] = []          # owner[k] = 该行属于 run 里的第几个块
    for bi, b in enumerate(run):
        for ln in b.text.splitlines():
            lines.append(ln)
            owner.append(bi)

    tables = find_tables(lines)
    if not tables:
        _flag_suspects(lines, parsed, report, excluded=[])
        return list(run)

    def _in_table(k: int) -> RecoveredTable | None:
        return next((t for t in tables if t.line_start <= k < t.line_end), None)

    out: list[Block] = []
    k = 0
    while k < len(lines):
        t = _in_table(k)
        if t is not None:
            out.append(_table_block(t.rows, page=run[owner[k]].page))
            report.recovered += 1
            k = t.line_end
            continue
        # 普通行：**按原始块边界**聚合（不把无关段落缝在一起）
        bi = owner[k]
        buf = [lines[k]]
        k += 1
        while k < len(lines) and owner[k] == bi and _in_table(k) is None:
            buf.append(lines[k])
            k += 1
        text = "\n".join(buf).strip()
        if text:
            out.append(Block(kind="paragraph", text=text, page=run[bi].page))

    _flag_suspects(lines, parsed, report, excluded=tables)
    return out


def _flag_suspects(lines: list[str], parsed: ParsedDoc, report: RecoveryReport,
                   excluded: list[RecoveredTable]) -> None:
    """宽判据：把"疑似表格但没重建"的区间**记下来**（文本一字不动）。"""
    for s in find_suspects(lines):
        if any(t.line_start <= s.line_start < t.line_end for t in excluded):
            continue                                  # 已在重建范围里，不重复记
        report.suspect.append(s)
    if report.suspect:
        preview = "；".join(
            f"第 {s.line_start}-{s.line_end} 行：{s.reason}" for s in report.suspect[:3]
        )
        _add_warning(parsed, f"疑似表格但未重建（{preview}）")


def _table_block(rows: list[list[str]], page: int | None) -> Block:
    """与 `parsers._table_block` 同构：自然语言版 + 原表 + 复杂度。

    （这里不 import `parsers`，避免两个模块互相依赖 —— 两者都只是调用 `tables` 的那套函数。）
    """
    return Block(
        kind="table",
        text=render_nl(rows, caption=None),
        page=page,
        raw_table=rows_to_markdown(rows),
        table_complex=is_complex(rows),
        meta={"recovered": True},   # 标记来源：它是**重建**的，不是解析器给的
    )


def _add_warning(parsed: ParsedDoc, msg: str) -> None:
    if msg not in parsed.warnings:
        parsed.warnings.append(msg)
