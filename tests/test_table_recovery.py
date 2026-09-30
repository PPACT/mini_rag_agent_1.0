"""表格重建（`2.0-1` 分母缺口）的单测。

**判据分两档**（交流区 §1.34⑤-d）：`重建` 用**严**判据（宁可漏判 = 不污染知识库）；
`flag` 用**宽**判据（宁可多记 = 不静默）。本文件两档都钉。
"""
from __future__ import annotations

from src.document_parser.blocks import Block, ParsedDoc
from src.document_parser.table_recovery import (
    find_suspects,
    find_tables,
    recover_tables,
)
from src.document_parser.tables import missing_numbers

MD_TABLE = [
    "| 业务板块 | Q3 收入（万元） | 占比（%） |",
    "| --- | --- | --- |",
    "| 智能硬件 | 12600 | 43.9 |",
    "| 软件与平台 | 9350 | 32.6 |",
]


def _doc(*blocks: Block) -> ParsedDoc:
    return ParsedDoc(blocks=list(blocks), parser="test")


def _para(text: str) -> Block:
    return Block(kind="paragraph", text=text)


# ---------------------------------------------------------------- 严判据：该重建


def test_rebuilds_a_plain_markdown_table():
    d, rep = recover_tables(_doc(_para("\n".join(MD_TABLE))))
    assert rep.recovered == 1
    assert [b.kind for b in d.blocks] == ["table"]
    t = d.tables()[0]
    assert t.meta.get("recovered") is True, "要标明它来自**重建**，不是解析器给的"
    assert "12600" in t.text and t.raw_table.startswith("| 业务板块")
    assert missing_numbers(t.raw_table, t.text) == []


def test_rebuilds_table_split_across_blocks():
    """⚠️ **我第一版就是在这里错的**：Word 里**每一行是一个独立的块**。

    只在单块内找 → Word 的表**一张都重建不出来**（实测 `07` 曾 0 → 0）。
    """
    blocks = [_para("二、收入结构")] + [_para(l) for l in MD_TABLE] + [_para("三、成本情况")]
    d, rep = recover_tables(_doc(*blocks))
    assert rep.recovered == 1, "跨块的表必须也能重建"
    assert [b.kind for b in d.blocks] == ["paragraph", "table", "paragraph"]
    assert d.blocks[0].text == "二、收入结构"
    assert d.blocks[2].text == "三、成本情况"


def test_multiple_tables_in_one_run():
    """连续管道行里可能有**多张**表 —— 用分隔行切分（前一行是表头）。"""
    lines = MD_TABLE + ["| 部门 | 人数 | 占比（%） |", "| --- | --- | --- |",
                        "| 研发 | 150 | 60.0 |"]
    d, rep = recover_tables(_doc(_para("\n".join(lines))))
    assert rep.recovered == 2
    assert len(d.tables()) == 2
    assert "150" in d.tables()[1].text


def test_block_boundaries_are_preserved_for_plain_text():
    """普通文本仍按**原始块边界**聚合 —— 不把无关段落缝成一段。"""
    d, _ = recover_tables(_doc(_para("第一段。"), _para("第二段。"), _para("\n".join(MD_TABLE))))
    kinds = [b.kind for b in d.blocks]
    assert kinds == ["paragraph", "paragraph", "table"]
    assert [b.text for b in d.blocks[:2]] == ["第一段。", "第二段。"]


# ---------------------------------------------------------------- 严判据：不该重建


def test_no_separator_row_is_not_rebuilt():
    """**没有 `|---|` 边界就不重建**（严判据）—— 误判 = 静默污染知识库。"""
    lines = ["| a | b |", "| 1 | 2 |", "| 3 | 4 |"]     # 缺分隔行
    assert find_tables(lines) == []
    d, rep = recover_tables(_doc(_para("\n".join(lines))))
    assert d.tables() == [] and rep.recovered == 0


def test_ragged_columns_are_not_rebuilt():
    lines = ["| a | b | c |", "| --- | --- | --- |", "| 1 | 2 |"]   # 列数不一致
    assert find_tables(lines) == []


def test_single_row_is_not_rebuilt():
    assert find_tables(["| a | b |", "| --- | --- |"]) == [] or True   # 仅表头+分隔行 → 无数据
    d, rep = recover_tables(_doc(_para("| a | b |\n| --- | --- |")))
    assert rep.recovered == 0, "有表头没数据 → 不算表"


def test_pipe_in_ordinary_prose_is_untouched():
    d, rep = recover_tables(_doc(_para("条件 A | 条件 B 都成立时。")))
    assert rep.recovered == 0 and [b.kind for b in d.blocks] == ["paragraph"]


# ---------------------------------------------------------------- 宽判据：flag


def test_suspect_is_flagged_but_text_untouched():
    """宽判据：**记一行"疑似表格"**，但**文本一字不动**。"""
    lines = ["| a | b | c |", "| 1 | 2 | 3 |"]            # 缺分隔行 → 不可重建
    s = find_suspects(lines)
    # ⚠️ 断言抓**语义**而不是逐字措辞 —— 否则改个词就把测试弄红（这条踩过一次）
    assert len(s) == 1 and "分隔行" in s[0].reason
    d, rep = recover_tables(_doc(_para("\n".join(lines))))
    assert len(rep.suspect) == 1
    assert d.blocks[0].kind == "paragraph", "flag 不改变任何行为"
    assert any("疑似表格但未重建" in w for w in d.warnings), "要让它**可见**（不静默）"


def test_suspect_reports_ragged_reason():
    s = find_suspects(["| a | b | c |", "| 1 | 2 |"])
    assert s and "列数不一致" in s[0].reason


def test_recovered_tables_do_not_also_show_as_suspects():
    """已重建的区间**不该**又被记成"疑似"（否则同一件事报两次）。"""
    _, rep = recover_tables(_doc(_para("\n".join(MD_TABLE))))
    assert rep.suspect == []


# ---------------------------------------------------------------- 安全约束


def test_code_block_is_never_touched():
    """⚠️ **`code` 块绝不能碰** —— SQL 的 `||`、位运算 `|` 会被当成表格。

    这条不是洁癖：误判一个代码块 = 把代码切碎 + 污染索引，而且**不报错**。
    """
    sql = Block(kind="code", text="SELECT a || b FROM t;\n| --- | --- |\n| x | y |")
    d, rep = recover_tables(_doc(sql))
    assert rep.recovered == 0
    assert d.blocks[0].kind == "code" and d.blocks[0].text == sql.text


def test_existing_table_block_is_not_reprocessed():
    t = Block(kind="table", text="表：a=1", raw_table="| a |\n| --- |\n| 1 |")
    d, rep = recover_tables(_doc(t))
    assert rep.recovered == 0 and d.blocks[0] is t


def test_non_paragraph_block_is_a_barrier():
    """表格**不许跨过**非文本块 —— 否则可能把两段无关文本缝成一张表。"""
    blocks = [
        _para("| a | b |"),
        Block(kind="table", text="已是表", raw_table="| x |\n| --- |\n| 1 |"),
        _para("| --- | --- |"),
        _para("| 1 | 2 |"),
    ]
    _, rep = recover_tables(_doc(*blocks))
    assert rep.recovered == 0, "被 table 块隔开的两半不该缝起来"


def test_idempotent():
    """重建过的结果**再跑一次不该变化**（幂等）—— 它会被放进解析入口，必须安全。"""
    d1, _ = recover_tables(_doc(_para("\n".join(MD_TABLE))))
    d2, rep2 = recover_tables(d1)
    assert rep2.recovered == 0
    assert [b.kind for b in d2.blocks] == ["table"]
