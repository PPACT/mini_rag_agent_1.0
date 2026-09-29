"""P0-1 的对外形态：判定 `unknown` → **答案照给 + 显式标注 + 反问用户**。

用户裁决的组合是「B + C」：
- B 的一半：**不拒答**，答案照给（判定器失败 ≠ 答案不存在）
- C 的一半：**把不确定性丢回给用户**，请其确认/补充
合起来 = **答了，但明确告诉你"这条没被确认过"，并问你指的是哪一份**。

要消灭的是**静默**（"没判定"被当成"判定了：不歧义"），不是把护栏失灵
升级成"用户拿不到答案"。
"""
from src.api.chat_api import _NO_ANSWER_TEXT, _annotate_unverified
from src.schemas.chat import ChatResponse
from src.vector_store.base import Chunk


def _chunk(source: str, cid: str = "x") -> Chunk:
    return Chunk(document_id="d", chunk_index=0, content="c", source_file=source, id=cid)


def test_annotation_keeps_the_answer():
    """**不是拒答**：答案必须还在。"""
    out = _annotate_unverified("年假五天。", "判定模型没有返回任何内容", [_chunk("员工手册.docx")])
    assert "年假五天。" in out


def test_annotation_says_why_it_is_unsure():
    out = _annotate_unverified("x", "判定模型输出被 max_tokens 截断", [_chunk("a.docx")])
    assert "没有得到确认" in out
    assert "max_tokens 截断" in out, "要说清为什么不确定，不能只给一句模糊话"


def test_annotation_asks_user_with_retrieved_sources():
    out = _annotate_unverified("x", "r", [_chunk("A.docx"), _chunk("B.docx")])
    assert "请指明" in out
    assert "A.docx" in out and "B.docx" in out


def test_annotation_wording_does_not_overclaim_candidates():
    """⚠️ 措辞必须如实：这些是**检索命中的来源**，**不是**"互相矛盾的候选答案"。

    判定器 failure 时它**没给出 `options`**，所以这里给不出真正的候选 ——
    措辞上不能暗示做了一个没做的判断。
    """
    out = _annotate_unverified("x", "r", [_chunk("A.docx")])
    assert "检索到这些来源" in out
    assert "候选" not in out


def test_annotation_dedupes_and_caps_sources():
    chunks = [_chunk(f"f{i}.docx", str(i)) for i in range(8)] + [_chunk("f0.docx", "dup")]
    out = _annotate_unverified("x", "r", chunks)
    assert out.count("f0.docx") == 1, "来源要去重"
    assert "f5.docx" not in out, "最多列 5 个，避免把答案淹没在文件清单里"


def test_annotation_without_sources_does_not_ask():
    out = _annotate_unverified("x", "r", [])
    assert "没有得到确认" in out
    assert "请指明" not in out, "没有可指明的来源就不该问"


def test_annotation_tolerates_empty_reason():
    out = _annotate_unverified("x", "", [_chunk("a.docx")])
    assert "未返回有效结论" in out


def test_no_answer_text_is_explicit():
    """生成空答案时给用户的必须是**明确说明**，不是空白。"""
    assert _NO_ANSWER_TEXT.strip()
    assert "没能" in _NO_ANSWER_TEXT


# ---- 响应契约 ----


def test_chat_response_defaults_to_clear():
    r = ChatResponse(answer="a", sources=[])
    assert r.judge_status == "clear"
    assert r.need_clarification is False


def test_judge_status_is_independent_of_need_clarification():
    """`unknown` 时 answer 是**真答案**、不是澄清话术 → 不能靠 `need_clarification` 表达。

    `need_clarification=True` 的契约是「answer **是澄清话术**」，
    拿它表示 unknown 会让前端把真答案当澄清话术渲染。
    """
    r = ChatResponse(answer="真答案", sources=[], judge_status="unknown")
    assert r.need_clarification is False
    assert r.judge_status == "unknown"


def test_ambiguous_clarify_contract_unchanged():
    """澄清路径的契约保持原样（answer 是澄清话术 + 候选列表）。"""
    r = ChatResponse(answer="请问你指的是哪一种？", sources=[], need_clarification=True,
                     judge_status="ambiguous")
    assert r.need_clarification is True
    assert r.judge_status == "ambiguous"
