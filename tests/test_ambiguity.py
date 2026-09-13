"""歧义判定模块的单元测试。"""
from src.rag.ambiguity import AmbiguityResult, is_diverse, parse_result
from src.vector_store.base import Chunk


def _chunk(source: str, cid: str = "x") -> Chunk:
    return Chunk(document_id="d", chunk_index=0, content="c", source_file=source, id=cid)


# ---- 门控信号 ----

def test_is_diverse_below_threshold():
    chunks = [_chunk("a.md", "1"), _chunk("a.md", "2"), _chunk("b.md", "3")]
    assert is_diverse(chunks, threshold=3) is False


def test_is_diverse_meets_threshold():
    chunks = [_chunk("a.md", "1"), _chunk("b.md", "2"), _chunk("c.md", "3")]
    assert is_diverse(chunks, threshold=3) is True


def test_is_diverse_empty():
    assert is_diverse([], threshold=3) is False


# ---- 解析与安全回退 ----

def test_parse_ambiguous_with_options():
    raw = '{"ambiguous": true, "reason": "时限不一致", "options": [{"source": "a.md", "summary": "90 天"}, {"source": "b.md", "summary": "93 天"}]}'
    r = parse_result(raw)
    assert r.ambiguous is True
    assert r.reason == "时限不一致"
    assert [o.source for o in r.options] == ["a.md", "b.md"]
    assert r.options[0].summary == "90 天"


def test_parse_not_ambiguous():
    r = parse_result('{"ambiguous": false, "reason": "", "options": []}')
    assert r == AmbiguityResult()


def test_parse_with_surrounding_text():
    raw = '判定如下：{"ambiguous": true, "reason": "冲突", "options": []} 完毕'
    r = parse_result(raw)
    assert r.ambiguous is True


def test_parse_invalid_returns_not_ambiguous():
    """解析失败必须安全回退为「无歧义」，避免误伤（动不动就反问）。"""
    assert parse_result("无法判定").ambiguous is False
    assert parse_result("").ambiguous is False
    assert parse_result("{不是合法 json}").ambiguous is False
    assert parse_result("[1,2,3]").ambiguous is False


def test_parse_skips_malformed_options():
    raw = '{"ambiguous": true, "reason": "x", "options": [{"source": "a.md"}, {"summary": "有效"}, "垃圾"]}'
    r = parse_result(raw)
    assert len(r.options) == 1
    assert r.options[0].summary == "有效"
