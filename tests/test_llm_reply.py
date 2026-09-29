"""P0-4：`LLMReply` 必须让「**模型没答**」与「**模型答了**」可区分。

原来 `complete()` 结尾是 `return msg.content or ""` —— **`finish_reason` 被丢弃**，
于是「被 `max_tokens` 截断」与「模型本来就没答」在代码里长得**一模一样**：
调用方只能拿到一个空串，然后**静默继续**。

P0-1 明确要求把 `finish_reason=length` 判为"无法判断" ——
**不把元信息带出来，那条要求根本实现不了**（这是清单里没点出的前置条件）。
"""
from src.config.litellm_client import LLMReply


def test_answered():
    assert LLMReply(text="有内容").answered is True
    assert LLMReply(text="").answered is False
    assert LLMReply(text="   \n ").answered is False, "纯空白不算答了"


def test_truncated_is_distinguishable_from_empty():
    """**核心断言**：同样是空文本，`length` 与 `stop` 必须能分开。"""
    truncated = LLMReply(text="", finish_reason="length")
    empty = LLMReply(text="", finish_reason="stop")
    assert truncated.truncated is True
    assert empty.truncated is False
    assert truncated.empty_reason == "truncated"
    assert empty.empty_reason == "empty"


def test_empty_reason_is_none_when_answered():
    assert LLMReply(text="ok", finish_reason="stop").empty_reason is None


def test_missing_finish_reason_is_tolerated():
    """有些 provider 不返回 finish_reason —— 不能因此炸，且要当成"没答"而非"截断"。"""
    r = LLMReply(text="")
    assert r.finish_reason is None
    assert r.truncated is False
    assert r.empty_reason == "empty"


def test_finish_reason_is_case_insensitive():
    assert LLMReply(text="", finish_reason="LENGTH").truncated is True
