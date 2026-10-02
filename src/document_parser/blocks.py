"""解析产物的统一中间表示（IR）—— 2.0 新增。

**为什么要有这一层**：PDF / Word / Excel / PPT 各有各的库与结构，
而下游（切块 / 入库 / 检索）只关心三件事 ——
「**一段可检索的文本**」「**它在第几页**」「**若它是表格，原表长什么样**」。
把这层定死，路由与切块就不必知道具体是哪个库解析出来的。

⚠️ 设计依据：`docs/local/参考资料/RAG 数据清洗、多格式录入与切块路由方案规划表.md`
§2.2「输出结构」列 + 交流区 §1.24 Q5 的表格口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# 段落 / 标题 / 表格 / 代码。
# ⚠️ 标题**单独一类**不是洁癖：切块级路由要按它做「章节递归切」，
#    混进 paragraph 就再也认不出来了。
#
# ⭐ `2.0-31`（**新增内容必须成「新的块」，不许并进已有块的 `content`**）：
#    将来加**图片描述**（`2.0-21`）这类新内容时，**必须**：
#      ① 在这里**新增一个 kind**（如 `"image"`）；
#      ② **同时**把它加进 `chunking._ATOMIC` —— 否则它会被并进 text 段，
#         与相邻段落共用一个 `content` → **那些块的内容变了 → 必须重嵌**。
#    ✅ 独立成块则**已有块一个字未变 → 零重嵌**。
#    ⚠️ **越晚越贵**：一旦有人先把描述并进 `content`，改回来要**重嵌全库**。
#    🚩 **闸**：`tests/test_chunking.py::test_new_kind_must_be_atomic`
#       —— 加了新 kind 却没进 `_ATOMIC` 时它会**直接红**。
BlockKind = Literal["paragraph", "heading", "table", "code"]


@dataclass
class Block:
    """解析产物的最小单元。"""

    kind: BlockKind
    text: str
    """送检索的文本。**表格这里放的是「自然语言版」**（见 2.0-1 口径）。"""

    page: int | None = None
    """1-based 页码。**不适用就是 None** —— 不用 0 / -1 假装"有页码"。

    来源：PDF 逐页解析；Word/Excel 没有稳定页码概念（分页由渲染决定）→ None。
    """

    raw_table: str | None = None
    """原表（Markdown）。仅 `kind == "table"` 有值。**只进词法、不进向量**（2.0-1 口径）。"""

    table_complex: bool = False
    """复杂表（嵌套表头 / 多级合并 / 列数不齐）→ 其**原表也进向量**，且让评测能分开看。

    ⚠️ 是"**这类**进"，不是全局都进 —— 别因为少数难表把所有表格的向量都污染了。
    """

    meta: dict = field(default_factory=dict)
    """解析器给的其他信息：`sheet`（Excel 表名）/ `slide`（PPT 序号）/ `lang`（代码语言）等。"""


@dataclass
class ParsedDoc:
    """一次解析的全部产物。"""

    blocks: list[Block]
    parser: str
    """**实际用上的**解析器名（如 `python-docx`）。

    ⚠️ 单独记它，是为了能核实「路由**打算**用什么」≠「**实际**用了什么」——
    协议 **P-2** 说这是本项目最高频的坑，而路由 + fallback 正是最容易漂移的地方。
    """

    pages: int | None = None
    """文档总页数（有页码信息时）。"""

    warnings: list[str] = field(default_factory=list)
    """解析过程中的**非致命**问题（如"某页无文本层"）—— 收集起来一并报，不吞掉。"""

    @property
    def text(self) -> str:
        """把所有 block 的文本拼成一份纯文本。

        给仍只吃文本的调用方（`load_text()`）用。
        ⚠️ 表格块的 `text` 是自然语言版 —— 所以**表格从 2.0 起不再被丢掉**。
        """
        return "\n".join(b.text for b in self.blocks if b.text)

    def tables(self) -> list[Block]:
        """本次解析出的表格块。"""
        return [b for b in self.blocks if b.kind == "table"]

    def kind_counts(self) -> dict[str, int]:
        """各类块的数量（汇总输出用，避免把明细刷进上下文）。"""
        counts: dict[str, int] = {}
        for b in self.blocks:
            counts[b.kind] = counts.get(b.kind, 0) + 1
        return counts
