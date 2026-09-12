"""LLMReranker 输出解析的单元测试。"""
from src.rag.reranker import LLMReranker


def test_parse_valid_order():
    assert LLMReranker._parse("[3,0,5,1,2,4]", 6) == [3, 0, 5, 1, 2, 4]


def test_parse_with_surrounding_text():
    assert LLMReranker._parse("排序结果如下：[2,0,1]", 3) == [2, 0, 1]


def test_parse_fills_missing_indices():
    # LLM 漏掉 1、2 → 按原序补到末尾
    assert LLMReranker._parse("[3,0]", 4) == [3, 0, 1, 2]


def test_parse_dedupes():
    assert LLMReranker._parse("[1,1,0]", 2) == [1, 0]


def test_parse_ignores_out_of_range():
    assert LLMReranker._parse("[9,0,1]", 2) == [0, 1]


def test_parse_invalid_returns_empty():
    assert LLMReranker._parse("无法排序", 3) == []
    assert LLMReranker._parse("", 3) == []
    assert LLMReranker._parse("[a,b]", 3) == []
