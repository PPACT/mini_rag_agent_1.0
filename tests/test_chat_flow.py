"""`/chat` 端点的**分支级**集成测试（P0-1 / P0-4）。

**为什么必须有它**：P0-A 交付时我明确标注过一处**没做端到端** ——
`/chat` 的 `unknown → 标注` 分支**跑不起来**：真实库是**有意留空的**，
而 `/chat` 固定查真库 → 取不到候选 → 走不到判定。
（`/demo` 已验证"三态能抵达响应"，但**标注逻辑只在 `/chat` 里**。）

所以这里把 `chat()` 的**外部依赖全部换成桩**（检索 / 判定 / 生成 / Redis），
但**保留它自己的分支逻辑不动** —— 测的就是"判定为 unknown 时，这段代码到底做了什么"。

⚠️ 与 `test_answer_degrade.py` 的分工：那里测 `_annotate_unverified` **函数本身**，
这里测**它在 `chat()` 里有没有被正确接上**。
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

from src.api import chat_api
from src.auth.deps import User
from src.rag.ambiguity import AmbiguityResult
from src.schemas.chat import ChatRequest, ChatResponse, ClarifyOption
from src.vector_store.base import Chunk

USER = User(name="测试用户", department="IT", secret_level=3)
QUESTION = "年假几天？"


class _FakeRedis:
    def __init__(self) -> None:
        self.stored: dict = {}

    async def get(self, key):
        return self.stored.get(key)

    async def set(self, key, value, ex=None):  # noqa: ARG002
        self.stored[key] = value


class _FakeAgent:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _payload):
        return {"messages": [AIMessage(content=self._content)]}


class _StubSettings:
    llm_api_key = "test-key"
    company_scope = "公司"
    top_k = 5
    cache_ttl = 60


def _chunk(source: str, cid: str = "c1") -> Chunk:
    return Chunk(document_id="d", chunk_index=0, content="年假五天。", source_file=source, id=cid)


def _wire(monkeypatch, *, answer="年假五天。", ambiguity=None, chunks=None):
    """把 `chat()` 的所有外部依赖换成桩；返回 (fake_redis, 捕获到的 ERR 事件名)。"""
    chunks = [_chunk("员工手册.docx")] if chunks is None else chunks
    fake_redis = _FakeRedis()
    errs: list[str] = []

    monkeypatch.setattr(chat_api, "get_settings", lambda: _StubSettings())
    monkeypatch.setattr(chat_api, "get_redis", lambda: fake_redis)
    monkeypatch.setattr(chat_api, "get_agent", lambda: _FakeAgent(answer))
    monkeypatch.setattr(chat_api, "load_templates", lambda: ("sys", "{context}||{question}"))
    monkeypatch.setattr(chat_api, "audit_err", lambda e, **kw: errs.append(e))

    async def _retrieve(question, departments, secret_level, **kw):  # noqa: ARG001
        return "检索上下文", chunks

    monkeypatch.setattr(chat_api, "retrieve", _retrieve)

    async def _ambiguity(question, ch):  # noqa: ARG001
        return ambiguity if ambiguity is not None else AmbiguityResult()

    monkeypatch.setattr(chat_api, "check_ambiguity", _ambiguity)
    return fake_redis, errs


def _ask():
    return asyncio.run(chat_api.chat(ChatRequest(question=QUESTION), USER))


# ---- unknown → 答案 + 标注 + 反问（P0-A 的核心分支）----


def test_unknown_annotates_answer_and_sets_status(monkeypatch):
    """**这是 P0-A 真正要验的那条路**：判定无有效结论时，代码到底做了什么。"""
    _, errs = _wire(monkeypatch, ambiguity=AmbiguityResult(
        status="unknown", reason="判定模型没有返回任何内容"))
    resp = _ask()

    assert resp.judge_status == "unknown"
    assert resp.need_clarification is False, "unknown 的 answer 是真答案，不是澄清话术"
    assert "年假五天。" in resp.answer, "**不能拒答**：答案必须还在"
    assert "没有得到确认" in resp.answer
    assert "判定模型没有返回任何内容" in resp.answer, "要说清为什么不确定"
    assert "请指明" in resp.answer, "要反问了用户"
    assert "answer_unverified" in errs, "必须记 ERR（P0-1 要求）"


def test_clear_answer_is_not_annotated(monkeypatch):
    """判 clear 时**不得**画蛇添足 —— 否则人人都被加上"未经确认"。"""
    _, errs = _wire(monkeypatch, ambiguity=AmbiguityResult(status="clear"))
    resp = _ask()

    assert resp.judge_status == "clear"
    assert resp.answer == "年假五天。"
    assert "没有得到确认" not in resp.answer
    assert errs == []


# ---- ambiguous → 澄清（契约不能动）----


def test_ambiguous_takes_clarify_path(monkeypatch):
    calls: list = []

    async def _build(question, result):  # noqa: ARG001
        calls.append(1)
        return "请问你指的是哪一种？"

    monkeypatch.setattr(chat_api, "build_clarification", _build)
    _wire(monkeypatch, ambiguity=AmbiguityResult(
        status="ambiguous", reason="时限不一致",
        options=[ClarifyOption(source="a.docx", summary="90 天")]))
    resp = _ask()

    assert resp.need_clarification is True
    assert resp.judge_status == "ambiguous"
    assert resp.answer == "请问你指的是哪一种？"
    assert [o.source for o in resp.clarify_options] == ["a.docx"]
    assert calls == [1]


# ---- 生成侧静默降级（P0-4）----


def test_empty_generation_is_explicit_not_blank(monkeypatch):
    """空答案**不再原样返回**（原来用户看到一片空白，日志无异常）。"""
    _, errs = _wire(monkeypatch, answer="   ", ambiguity=AmbiguityResult())
    resp = _ask()

    assert resp.answer == chat_api._NO_ANSWER_TEXT
    assert resp.answer.strip(), "不能是空串"
    assert "answer_empty" in errs


def test_empty_answer_takes_precedence_over_annotation(monkeypatch):
    """两者**互斥**：没有答案就谈不上"标注不确定"。

    否则会给一段"我没能生成答案"的话，再加一句"这条答案未经确认" —— 自相矛盾。
    """
    _, errs = _wire(monkeypatch, answer="",
                    ambiguity=AmbiguityResult(status="unknown", reason="判定没答"))
    resp = _ask()

    assert resp.answer == chat_api._NO_ANSWER_TEXT
    assert "没有得到确认" not in resp.answer
    assert errs == ["answer_empty"], "不该同时记两条"


# ---- 缓存 ----


def test_result_is_cached(monkeypatch):
    fake, _ = _wire(monkeypatch, ambiguity=AmbiguityResult(status="unknown", reason="x"))
    _ask()
    assert len(fake.stored) == 1


def test_cache_hit_short_circuits_retrieval(monkeypatch):
    """命中缓存时不该再走检索（这是缓存存在的意义）。"""
    fake, _ = _wire(monkeypatch)
    calls: list = []

    async def _retrieve(*a, **kw):  # noqa: ARG001
        calls.append(1)
        return "ctx", []

    monkeypatch.setattr(chat_api, "retrieve", _retrieve)
    key = chat_api._cache_key(QUESTION, USER.department, USER.secret_level,
                              _StubSettings.top_k, "real")
    fake.stored[key] = ChatResponse(answer="缓存里的答案", sources=[]).model_dump_json()

    assert _ask().answer == "缓存里的答案"
    assert calls == [], "命中缓存仍去检索了"
