"""`format_context` 的口径：**`raw_table` 只对「复杂表」放行**（交流区 §1.25②）。

三条去向必须分清（**别混成一条**）：

| 去向 | 一般表 | 复杂表（`table_complex`） |
|---|---|---|
| **送 LLM 的 context** | ❌ 不带 | ✅ **带**（这类表恰恰"可能不保真"，原表是必要兜底） |
| **随 `Source` 回前端**（给人核对） | ✅ 带 | ✅ 带 |
| 词法索引 | ✅ 进 | ✅ 进 |

**为什么不给一般表也塞**：`content` 是自然语言版，受「原表每个数值都要能在自然语言版里找到」
这条验收管着（`tests/test_tables.py` 里有守卫 + 反向用例）。保真既已保证，再塞一遍只是占 token。
"""
from __future__ import annotations

from src.rag.retriever import format_context
from src.vector_store.base import Chunk


def _chunk(*, content: str, raw_table: str | None = None, complex_: bool = False) -> Chunk:
    return Chunk(document_id="d", chunk_index=0, content=content,
                 source_file="手册.docx", raw_table=raw_table, table_complex=complex_)


def test_plain_table_is_not_sent_to_llm():
    """一般表：**只给自然语言版**（保真已由验收守住）。"""
    ctx = format_context([_chunk(content="表：城市等级=一线城市；住宿上限=600",
                                 raw_table="| 城市等级 | 住宿上限 |\n| --- | --- |\n| 一线城市 | 600 |")])
    assert "住宿上限=600" in ctx
    assert "| 城市等级 |" not in ctx, "一般表的原表不该进 LLM context"


def test_complex_table_is_sent_to_llm():
    """复杂表：**原表也要给** —— 它可能不保真，原表是兜底。"""
    raw = "| 差旅标准 |  | 备注 |\n| --- | --- | --- |\n| 一线 | 600 | - |"
    ctx = format_context([_chunk(content="表：差旅标准=一线；列2=600",
                                 raw_table=raw, complex_=True)])
    assert "【原表】" in ctx
    assert "| 差旅标准 |" in ctx, "复杂表的原表必须进 context（它是保真的兜底）"


def test_complex_flag_without_raw_table_is_harmless():
    """标记为复杂但没有原表（例如非表格块被误标）→ 不能崩、不能凭空造出【原表】。"""
    ctx = format_context([_chunk(content="正文。", complex_=True)])
    assert "正文。" in ctx
    assert "【原表】" not in ctx


def test_citation_numbering_survives_the_extra_block():
    """⚠️ 原表**必须留在同一个 `[来源N]` 块内** —— 否则引用编号会被打乱，
    而 `_cited_chunks` 靠 `[来源N]` 解析"LLM 到底引用了哪几条"。"""
    a = _chunk(content="甲", raw_table="| x |\n| --- |\n| 1 |", complex_=True)
    b = _chunk(content="乙")
    ctx = format_context([a, b])
    assert ctx.count("[来源1]") == 1, "来源1 只该出现一次（原表要并进同一块）"
    assert "[来源2]" in ctx
    assert ctx.index("[来源1]") < ctx.index("【原表】") < ctx.index("[来源2]"), \
        "原表必须落在 来源1 与 来源2 之间，不能跑到下一块后面"
