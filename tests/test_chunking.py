"""切块级路由（2.0-2 第 3 层）的测试 —— 直接构造 `ParsedDoc`，不必造文件。"""
from __future__ import annotations

from src.document_parser.blocks import Block, ParsedDoc
from src.document_parser.chunking import split_blocks


def _doc(*blocks: Block) -> ParsedDoc:
    return ParsedDoc(blocks=list(blocks), parser="test")


def _para(text: str, page: int | None = None) -> Block:
    return Block(kind="paragraph", text=text, page=page)


def _table(nl: str, raw: str, page: int | None = None, complex_: bool = False) -> Block:
    return Block(kind="table", text=nl, raw_table=raw, page=page, table_complex=complex_)


# ---- 表格必须整块存活 ----


def test_table_survives_as_one_chunk():
    """**本层存在的首要理由**：表格不能被从中间切断。

    切坏了会怎样：`住宿上限=600` 变成裸的 `600` —— **数值还在、语义没了**，
    而且这种坏法**不会报错**。
    """
    long_table = _table(
        "表：城市等级=一线城市，住宿上限=600；城市等级=二线城市，住宿上限=400；",
        "| 城市等级 | 住宿上限 |\n| --- | --- |\n| 一线城市 | 600 |\n| 二线城市 | 400 |",
    )
    chunks = split_blocks(_doc(long_table), chunk_size=10, overlap=2)  # 故意把 chunk_size 压到极小
    assert len(chunks) == 1, "表格必须整块，chunk_size 再小也不切"
    assert chunks[0].kind == "table"
    assert chunks[0].raw_table and "600" in chunks[0].raw_table
    assert chunks[0].text == long_table.text


def test_table_carries_metadata():
    chunks = split_blocks(_doc(_table("表：a=1", "| a |\n| --- |\n| 1 |", page=3, complex_=True)))
    assert chunks[0].page == 3
    assert chunks[0].table_complex is True


def test_code_block_is_atomic_too():
    code = Block(kind="code", text="def f():\n    return 1\n" * 30)
    chunks = split_blocks(_doc(code), chunk_size=10, overlap=2)
    assert len(chunks) == 1 and chunks[0].kind == "code"


# ---- 段落沿用原行为，且保持章节上下文 ----


def test_long_paragraph_is_still_split():
    """段落**不能**因为引入块级切块就变成整块 —— 那是把切块层做没了。"""
    para = _para("这是一句。" * 200)
    chunks = split_blocks(_doc(para), chunk_size=100, overlap=10)
    assert len(chunks) > 1, "长段落仍要按句切"
    assert all(c.kind == "paragraph" for c in chunks)


def test_heading_context_survives_across_a_table():
    """⚠️ 这条是**设计约束的回归测试**：不能让表格把标题与后文割开。

    `split_text` 靠跨段维护"当前章节"；按块分别调用会丢掉它 ——
    于是"表格之后那段"会失去章节归属。
    """
    chunks = split_blocks(_doc(
        Block(kind="heading", text="差旅制度"),
        _para("本制度适用于全体员工。"),
        _table("表：城市等级=一线城市，住宿上限=600", "| 城市等级 | 住宿上限 |\n| --- | --- |\n| 一线城市 | 600 |"),
        _para("报销须附行程单。"),
    ))
    after_table = [c for c in chunks if "报销须附行程单" in c.text]
    assert after_table, "表格之后的段落不能丢"
    assert after_table[0].title == "差旅制度", "表格之后的段落必须仍知道自己在哪一章节下"


# ---- 页边界断段：page 必须精确 ----


def test_page_boundary_breaks_the_segment():
    """⚠️ 若把多页并成一段，`page` 只能取首块 → **后面几页的内容被标上第一页的页码**。

    那是一个**看起来可信、其实编造**的页码 —— 比没有页码更糟（引用溯源会撒谎）。
    """
    chunks = split_blocks(_doc(
        _para("第一页的内容。", page=1),
        _para("第二页的内容。", page=2),
        _para("第三页的内容。", page=3),
    ))
    by_text = {c.text: c.page for c in chunks}
    assert by_text["第一页的内容。"] == 1
    assert by_text["第二页的内容。"] == 2
    assert by_text["第三页的内容。"] == 3


def test_same_page_paragraphs_may_merge():
    """同一页内的段落**可以**合并 —— 断段只为保证 page 精确，不是为了拆碎。"""
    chunks = split_blocks(_doc(_para("甲。", page=1), _para("乙。", page=1)))
    assert len(chunks) == 1
    assert "甲。" in chunks[0].text and "乙。" in chunks[0].text
    assert chunks[0].page == 1


# ---- 偏移语义 ----


def test_offsets_are_relative_to_the_segment():
    """2.0 起偏移**相对于所属段**（不再是"相对于整份文档拼接文本"，那在本层已无意义）。"""
    chunks = split_blocks(_doc(_para("甲乙丙丁。", page=1)))
    assert chunks[0].start == 0
    assert chunks[0].end == len(chunks[0].text)


def test_empty_document_yields_nothing():
    assert split_blocks(_doc()) == []


# ---- 2.0-46：紧邻 heading 的表格，heading 作 caption ----


def test_heading_before_table_becomes_caption():
    """`2.0-46`：**紧邻**（块序号差 1）的 heading + table → heading 作 caption，不再单独成块。"""
    chunks = split_blocks(_doc(
        Block(kind="heading", text="二、收入结构"),
        _table("表：业务板块=智能硬件；Q3 收入（万元）=12600",
               "| 业务板块 | Q3 收入 |\n| --- | --- |\n| 智能硬件 | 12600 |"),
    ))
    assert len(chunks) == 1, "heading 不该再单独成块（那是废块）"
    assert chunks[0].kind == "table"
    assert "表「二、收入结构」：" in chunks[0].text, "heading 要变成表格的 caption"
    assert chunks[0].title == "二、收入结构"


def test_caption_does_not_break_hierarchy():
    """⚠️ **关键**：Caption 化的 heading **仍然是章节标记** —— 后续块要能继承它。

    这正是"落点放切块层、不放解析层"的理由（文档侧 §1.37②）：
    在解析层吞掉 heading 会**丢层级**。
    """
    chunks = split_blocks(_doc(
        Block(kind="heading", text="二、收入结构"),
        _table("表：a=1", "| a |\n| --- |\n| 1 |"),
        _para("上表说明收入结构。"),
    ))
    after = [c for c in chunks if "上表说明收入结构" in c.text]
    assert after and after[0].title == "二、收入结构", "被吞的 heading 仍要喂给后续块"


def test_paragraph_between_heading_and_table_blocks_the_merge():
    """**中间夹了任何块 → 不合并**（判据是结构性的、且**宁可漏判**）。"""
    chunks = split_blocks(_doc(
        Block(kind="heading", text="二、收入结构"),
        _para("先说明一句。"),
        _table("表：a=1", "| a |\n| --- |\n| 1 |"),
    ))
    table = [c for c in chunks if c.kind == "table"][0]
    assert "表「" not in table.text, "中间夹了段落 → 不该把标题当 caption（误判会污染表格语义）"


def test_plain_paragraph_before_table_is_not_a_caption():
    """**普通段落**紧邻表格 → **不合并**（只有 heading 才算 caption）。"""
    chunks = split_blocks(_doc(_para("这是一句正文。"),
                               _table("表：a=1", "| a |\n| --- |\n| 1 |")))
    assert "表「" not in [c for c in chunks if c.kind == "table"][0].text


# ---------------------------------------------------------------- 2.0-31


def test_new_kind_must_be_atomic():
    """⭐ `2.0-31` 的**闸**：**新增的块类型必须同时进 `chunking._ATOMIC`**。

    ⚠️ **为什么这是一条真闸**：没进 `_ATOMIC` 的新块会被并进 text 段，
    和相邻段落共用同一个 `content` → **那些块的内容变了 → 必须重嵌**。
    （`2.0-30` 的 `content_hash` 事后能照出来，但那时已经晚了。）

    → **加了新 kind（比如 `2.0-21` 的 `image`）却没改 `_ATOMIC`，这条会直接红。**
    """
    from typing import get_args

    from src.document_parser.blocks import BlockKind
    from src.document_parser.chunking import _ATOMIC

    text_kinds = {"paragraph", "heading"}   # 这两类**本来**就是文本流，允许被合并
    missing = [k for k in get_args(BlockKind) if k not in text_kinds and k not in _ATOMIC]
    assert not missing, (
        f"这些块类型没进 `_ATOMIC`，会被并进 text 段、污染相邻块的 content：{missing}。"
        f"→ 新 kind 要「成新块」就必须同时加进 `chunking._ATOMIC`（2.0-31）"
    )


def test_atomic_insert_splits_the_text_segment():
    """⚠️ `2.0-31` 的**代价**：插入原子块会**切断它所在的 text 段**。

    ⚠️ 这条**记录事实，不是期望行为** —— 它纠正一句容易说过头的话：

    > 「独立成新块 → **已有块一个字未变** → **零重嵌**」

    **实测不成立**：不插时，两段短正文被 `split_text` **合并成 1 块**；
    插了原子块之后，`_segments` 在原子块处 `flush()` → 变成 **2 块** + 1 个原子块。
    → **被切断的那一段会重新切、要重嵌**（不是零）。

    ✅ **但约束本身仍是对的**：新内容**不许并进已有块的 `content`**
    （那会让那个块变成"夹着生成内容的杂块"，且**改回来要重嵌全库**）。
    ⚠️ 真做 `2.0-21` 时：**要么接受这段代价，要么先把"图片块放哪"想清楚再动手。**
    """
    before = _doc(_para("第一段正文，足够长以便自成一个块。"), _para("第二段正文。"))
    after = _doc(_para("第一段正文，足够长以便自成一个块。"),
                 Block(kind="code", text="print(1)"),
                 _para("第二段正文。"))

    n_before = [c.text for c in split_blocks(before)]
    n_after = [c.text for c in split_blocks(after)]

    # 事实一：不插时两段被合并（这正是"切断"会造成变化的根源）
    assert len(n_before) == 1 and "\n" in n_before[0], f"预期两段被合并，实际 {n_before}"
    # 事实二：插了之后被切成两块 —— ⚠️ 也就是"零重嵌"这句话在这里**不成立**
    assert len(n_after) == 3, f"预期被切成 3 块，实际 {n_after}"
    # 事实三：新块**独立成块**（没被并进邻居）—— 这是 `_ATOMIC` 的功劳
    assert "print(1)" in {c.text for c in split_blocks(after) if c.kind == "code"}
