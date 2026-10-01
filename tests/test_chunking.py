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
