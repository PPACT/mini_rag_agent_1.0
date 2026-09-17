"""引用校验的单元测试。"""
from src.api.chat_api import _cited_chunks
from src.vector_store.base import Chunk


def _chunks(n: int):
    return [Chunk(document_id="d", chunk_index=i, content=f"内容{i}", source_file=f"f{i}.md", id=str(i)) for i in range(n)]


def test_cites_subset():
    chunks = _chunks(5)
    cited = _cited_chunks(chunks, "答案是 A [来源2]，还有 B [来源5]。")
    assert [c.chunk_index for c in cited] == [1, 4]  # 0-based


def test_cites_keeps_context_order():
    chunks = _chunks(5)
    cited = _cited_chunks(chunks, "引用 [来源5] 和 [来源2]。")
    assert [c.chunk_index for c in cited] == [1, 4]  # 按上下文顺序，不是答案出现顺序


def test_no_citation_falls_back_to_all():
    chunks = _chunks(3)
    cited = _cited_chunks(chunks, "这里没有任何引用。")
    assert len(cited) == 3  # 保守回退


def test_out_of_range_citation_ignored():
    chunks = _chunks(3)
    cited = _cited_chunks(chunks, "引用 [来源9]。")
    assert len(cited) == 3  # 9 超界 → 视为无有效引用 → 回退


def test_chinese_format():
    chunks = _chunks(4)
    cited = _cited_chunks(chunks, "按来源3的说法。")
    assert [c.chunk_index for c in cited] == [2]


def test_bare_number_in_body_not_parsed_as_citation():
    """答案正文里的裸数字（如"2 个工作日"）不得被误当成引用编号。"""
    chunks = _chunks(5)
    cited = _cited_chunks(chunks, "根据 [来源1]，须提前 2 个工作日申请。")
    assert [c.chunk_index for c in cited] == [0]  # 只 [来源1]，不含裸 2
