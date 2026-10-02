"""`2.0-50` 可见化那一步的闸：**语料 vs 库的差集**。

⚠️ 这个脚本存在的意义是"**让静默丢数据变成可见**"——
所以它的判据本身**也必须能失败**：语料里少一份、或库里多一份，都得红。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.check_ingest_complete import compare  # noqa: E402


def test_identical_is_clean():
    """反向用例的**底线**：完全一致时两个差集都空 —— 否则它会永远红。"""
    assert compare({"a.pdf", "b.docx"}, {"a.pdf", "b.docx"}) == {"missing": [], "unexpected": []}


def test_detects_a_dropped_file():
    """⭐ 主用例：语料里有、库里没有 = **丢了**（就是 `2.0-50` 要曝光的那个）。"""
    r = compare({"a.pdf", "b.pdf", "c.pdf"}, {"a.pdf", "c.pdf"})
    assert r["missing"] == ["b.pdf"], f"少了一份却报告没丢：{r}"


def test_detects_an_extra_document():
    """⚠️ **多出来的也要报** —— 库里混进语料目录之外的文件，同样会让评估数字解释不通。"""
    r = compare({"a.pdf"}, {"a.pdf", "old_draft.pdf"})
    assert r["unexpected"] == ["old_draft.pdf"], f"多出来的没报：{r}"
    assert r["missing"] == []


def test_both_directions_at_once():
    r = compare({"a.pdf", "b.pdf"}, {"b.pdf", "z.pdf"})
    assert r == {"missing": ["a.pdf"], "unexpected": ["z.pdf"]}
