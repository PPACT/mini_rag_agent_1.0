"""歧义判定模块的单元测试（含 P0-1 三态 / P1-3 移除门控）。"""
import asyncio

from src.config.litellm_client import LLMReply
from src.rag import ambiguity as amb
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


def test_parse_invalid_is_unknown_not_clear():
    """⚠️ **P0-1 反转了旧行为**（本用例的前身断言的是"解析失败 → 无歧义"）。

    旧行为的危害：把「**模型没答**」当成「**模型答了：不歧义**」→
    系统认为问题不歧义、拿互相矛盾的候选硬答，**且全程不报错**。

    ⚠️ 注意：旧断言（`.ambiguous is False`）**现在依然会通过** ——
    因为 `unknown.ambiguous` 也是 False。所以那条测试若不改写，就会变成
    「**看起来还绿、但不再验证它声称的东西**」。故这里改为直接断言 `status`。
    """
    for raw in ("无法判定", "", "{不是合法 json}", "[1,2,3]", '{"reason": "缺 ambiguous 键"}'):
        r = parse_result(raw)
        assert r.status == "unknown", f"{raw!r} 应判 unknown，实际 {r.status}"
        assert r.ambiguous is False, "unknown 不能算作 ambiguous"


def test_parse_explicit_false_is_clear():
    """模型**明确答了"不矛盾"** → 这才是 clear（与 unknown 严格区分）。"""
    r = parse_result('{"ambiguous": false, "reason": "候选一致", "options": []}')
    assert r.status == "clear"
    assert not r.unknown


def test_parse_missing_key_is_unknown():
    """答了 JSON 但没按约定给 `ambiguous` 键 → 不算"答了"（模型没遵守契约）。"""
    assert parse_result('{"reason": "我看了下"}').status == "unknown"


def test_unknown_is_not_ambiguous():
    """两个属性不能混用：`unknown` ≠ `ambiguous`。"""
    u = AmbiguityResult(status="unknown", reason="模型没答")
    assert u.unknown is True
    assert u.ambiguous is False


# ---- 三态：check_ambiguity 的四条 unknown 来源 ----


def _stub_complete(monkeypatch, reply=None, exc=None, calls=None):
    async def _fake(messages, temperature=0.1):  # noqa: ARG001
        if calls is not None:
            calls.append(messages)
        if exc is not None:
            raise exc
        return reply

    monkeypatch.setattr(amb, "complete_with_meta", _fake)


def test_call_failure_is_unknown(monkeypatch):
    """调用异常 → unknown。旧实现返回"无歧义"：**一次网络抖动就等于"判定通过"**。"""
    _stub_complete(monkeypatch, exc=RuntimeError("connection reset"))
    r = asyncio.run(amb.check_ambiguity("q", [_chunk("a.md"), _chunk("b.md")]))
    assert r.status == "unknown"
    assert "connection reset" in r.reason


def test_truncated_reply_is_unknown(monkeypatch):
    """`finish_reason=length` → unknown（这正是 P0-1 点名的那条，靠 finish_reason 才认得出）。"""
    _stub_complete(monkeypatch, reply=LLMReply(text="", finish_reason="length"))
    r = asyncio.run(amb.check_ambiguity("q", [_chunk("a.md"), _chunk("b.md")]))
    assert r.status == "unknown"
    assert "截断" in r.reason


def test_empty_reply_is_unknown(monkeypatch):
    _stub_complete(monkeypatch, reply=LLMReply(text="   ", finish_reason="stop"))
    r = asyncio.run(amb.check_ambiguity("q", [_chunk("a.md"), _chunk("b.md")]))
    assert r.status == "unknown"


def test_unparseable_reply_is_unknown(monkeypatch):
    _stub_complete(monkeypatch, reply=LLMReply(text="我觉得差不多吧"))
    r = asyncio.run(amb.check_ambiguity("q", [_chunk("a.md"), _chunk("b.md")]))
    assert r.status == "unknown"


def test_disabled_or_too_few_chunks_is_clear(monkeypatch):
    """**判定按设计未运行** → 归 clear（若归 unknown，一关开关全站就"无法判断"）。"""

    class _S:
        ambiguity_check_enabled = False
        ambiguity_snippet_chars = 300

    monkeypatch.setattr(amb, "get_settings", lambda: _S())
    assert asyncio.run(amb.check_ambiguity("q", [_chunk("a"), _chunk("b")])).status == "clear"


# ---- P1-3：门控已移除 ----


def test_gate_removed_calls_llm_even_for_single_source(monkeypatch):
    """**P1-3 回归**：来源只有 1 个时，老门控会直接跳过判定；现在必须**真的调 LLM**。

    否则移除门控就只是删了几行，行为没变 —— 而那样"漏报减少"也无从发生。
    """
    calls: list = []
    _stub_complete(monkeypatch,
                   reply=LLMReply(text='{"ambiguous": true, "reason": "冲突", "options": []}'),
                   calls=calls)
    same_source = [_chunk("same.md", "1"), _chunk("same.md", "2")]
    assert is_diverse(same_source, 3) is False, "前提：老门控会拦下这组"
    r = asyncio.run(amb.check_ambiguity("q", same_source))
    assert calls, "门控已移除：来源单一也必须调 LLM 判定"
    assert r.status == "ambiguous"


def test_parse_skips_malformed_options():
    raw = '{"ambiguous": true, "reason": "x", "options": [{"source": "a.md"}, {"summary": "有效"}, "垃圾"]}'
    r = parse_result(raw)
    assert len(r.options) == 1
    assert r.options[0].summary == "有效"
