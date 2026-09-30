"""表格处理单测 —— 其中 `test_numbers_survive_rendering` 就是 **2.0-1 的验收闸**。

口径（交流区 §1.24 Q5）：`content` = 自然语言版（进向量），`raw_table` = 原表（只进词法）。
**验收**：原表里的**每一个数值**都必须在自然语言版里被找到。
"""
from __future__ import annotations

import pytest

from src.document_parser.tables import (
    is_complex,
    missing_numbers,
    number_tokens,
    render_nl,
    rows_to_markdown,
)

REIMBURSE = [
    ["城市等级", "住宿上限（元/晚）", "备注"],
    ["一线城市", "600", "超出自理"],
    ["二线城市", "400", ""],
    ["其他城市", "300", "含税"],
]


# ---- 2.0-1 的验收：数值保真 ----


def test_numbers_survive_rendering():
    """**2.0-1 的验收闸**：原表每个数值都要能在自然语言版里找到。"""
    nl = render_nl(REIMBURSE, caption="差旅住宿标准")
    assert missing_numbers(rows_to_markdown(REIMBURSE), nl) == []


def test_fidelity_holds_for_awkward_numbers():
    """含千分位 / 小数 / 百分号 / 日期 / 负数 —— 这些最容易在渲染时被"顺手美化"掉。"""
    rows = [
        ["项目", "2024 年", "2025 年", "增幅"],
        ["营收", "1,234.50", "2,000", "62%"],
        ["净利", "-300", "0.5", "-101%"],
    ]
    nl = render_nl(rows, caption="财务摘要")
    assert missing_numbers(rows_to_markdown(rows), nl) == []


def test_raw_table_adds_no_unique_tokens():
    """⭐ **可观测的健康指标**（文档侧 §1.25② 的建议，比"改注释"更进一步）。

    `raw_table` 给词法索引带来的**独有 token 必须为空**：

    - **为空** → 说明**自然语言版是保真的**（转换做对了）；
    - **不为空** → **自然语言版有损**（丢了列 / 丢了值）→ **去查 `render_nl`**，别去查索引。

    为什么要有这条：口径原本给「原表进词法」的理由是"多一条精确召回路径"，
    而**实测那个理由是空的**（因为保真成立）。与其删掉理由，不如**把它变成断言** ——
    这样它从"一句解释"升级成"**下一轮改动会被它拦住**"。
    （同 D9④⑤⑥ 的教训：注释解释了不存在的事，要修；而**能失败的东西才算守卫**。）
    """
    from src.document_parser.tokenizer import tokenize

    shapes = [
        REIMBURSE,                                               # 普通表
        [["部门", "", "金额"], ["IT", "差旅", "600"]],            # 表头有合并（空单元格）
        [["季度", "金额"], ["Q3", "1,234.50"], ["Q4", "-300"]],  # 千分位 / 负数
        [["项目"], ["缺列的表"]],                                  # 单列
    ]
    for rows in shapes:
        nl = render_nl(rows, caption="t")
        extra = sorted(set(tokenize(nl + "\n" + rows_to_markdown(rows)).split())
                       - set(tokenize(nl).split()))
        assert not extra, (
            f"自然语言版可能**不保真**：raw_table 带来独有 token {extra}。"
            f"→ 去查 render_nl（不是索引的问题）"
        )


def test_health_indicator_can_actually_fail():
    """反向验证：上面那条指标**必须能失败** —— 否则它只是永远绿的摆设。

    用一份"有损的自然语言版"（故意漏掉一列）来触发它。
    """
    from src.document_parser.tokenizer import tokenize

    rows = [["季度", "金额"], ["Q3", "5000"]]
    lossy_nl = "表：季度=Q3"                       # 故意丢了"金额=5000"
    extra = set(tokenize(lossy_nl + "\n" + rows_to_markdown(rows)).split()) - set(tokenize(lossy_nl).split())
    assert extra, "有损的自然语言版必须让指标非空（否则这道闸是假的）"


def test_fidelity_gate_can_actually_fail():
    """⚠️ **反向验证**：这道闸必须能失败 —— 否则它只是永远绿的摆设。

    故意造一个"渲染时丢了数值"的场景（用简化渲染函数模拟）。
    """
    raw = rows_to_markdown(REIMBURSE)

    def _lossy(_rows):  # 模拟"顺手把数值省略掉"的坏渲染
        return "表：住宿标准见附件"

    lost = missing_numbers(raw, _lossy(REIMBURSE))
    assert "600" in lost and "400" in lost and "300" in lost


def test_number_normalization_ignores_thousand_separator():
    """`1,000` 与 `1000` 是同一个数 —— 否则会报一堆假缺失。"""
    raw = "| 金额 |\n| --- |\n| 1,000 |"
    assert missing_numbers(raw, "金额=1000") == []
    assert number_tokens("1,000 与 1000") == ["1000"]


# ---- 渲染本身 ----


def test_render_keeps_header_semantics():
    """每行都要带列名 —— 裸值 `600` 对检索没有意义，`住宿上限=600` 才有。"""
    nl = render_nl(REIMBURSE)
    assert "城市等级=一线城市" in nl
    assert "住宿上限（元/晚）=600" in nl
    assert "一线城市" in nl and "二线城市" in nl


def test_render_does_not_use_ellipsis():
    """不得用"同上/略"之类的省略 —— 省略等于丢信息，且丢得静默。"""
    nl = render_nl(REIMBURSE)
    assert "同上" not in nl and "略" not in nl
    # 每一行都要完整出现（3 行数据 → 至少 3 个"城市等级="）
    assert nl.count("城市等级=") == 3


def test_render_placeholder_for_missing_header():
    """表头缺失时用「列N」占位 —— 宁可承认不知道列名，也不编一个。"""
    nl = render_nl([["", ""], ["a", "b"]])
    assert "列1=a" in nl and "列2=b" in nl


def test_render_single_row_table_is_not_lost():
    """只有表头、没有数据行的表也要产出东西（不能渲染成空串）。"""
    nl = render_nl([["指标", "值"]])
    assert nl.strip() and "指标" in nl


def test_render_empty():
    assert render_nl([]) == ""


# ---- 原表 Markdown ----


def test_markdown_escapes_pipes():
    """单元格里的 `|` 必须转义 —— 否则整表列错位，而**词法仍能命中数值**，很晚才发现。"""
    md = rows_to_markdown([["表达式", "结果"], ["a|b", "1"]])
    assert "a\\|b" in md, "单元格里的 | 必须转义，否则整表列错位"
    lines = md.splitlines()
    assert len(lines) == 3, "表头 + 分隔行 + 1 数据行"
    # 每行都该是规整的表格行：**列错位会让行首/行尾不再是 `| ` / ` |`**
    # （比数竖线个数更结实 —— 数个数得先知道正确答案是多少，那是猜）
    for line in lines:
        assert line.startswith("| ") and line.endswith(" |"), line


def test_markdown_pads_ragged_rows():
    md = rows_to_markdown([["a", "b", "c"], ["1"]])
    assert md.splitlines()[2].count("|") == 4  # | 1 |  |  |


# ---- 复杂度判定 ----


def test_complex_when_ragged():
    assert is_complex([["a", "b", "c"], ["1"]]) is True


def test_complex_when_header_has_empty_cell():
    """Excel 合并单元格只有锚点有值 → 表头出现空格。"""
    assert is_complex([["部门", "", "金额"], ["IT", "差旅", "600"]]) is True


def test_not_complex_for_normal_table():
    assert is_complex(REIMBURSE) is False


def test_complex_empty():
    assert is_complex([]) is False


@pytest.mark.parametrize("text,expect", [("", []), ("没有数字", []), ("共 3 项，计 1.5 万", ["3", "1.5"])])
def test_number_tokens(text, expect):
    assert number_tokens(text) == expect
