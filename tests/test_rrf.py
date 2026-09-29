"""RRF 融合的单元测试。"""
from src.rag.retriever import rrf_merge
from src.vector_store.base import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(document_id="doc", chunk_index=0, content=f"内容{cid}", id=cid)


def test_single_list_order_preserved():
    lst = [_chunk("a"), _chunk("b"), _chunk("c")]
    merged = rrf_merge([lst], k=60)
    assert [c.id for c in merged] == ["a", "b", "c"]


def test_hit_in_multiple_lists_ranks_higher():
    # b 在两路都出现（且排名靠前），应排第一
    list1 = [_chunk("a"), _chunk("b")]
    list2 = [_chunk("b"), _chunk("c")]
    merged = rrf_merge([list1, list2], k=60)
    assert merged[0].id == "b"


def test_deduplicates_by_id():
    list1 = [_chunk("a"), _chunk("b")]
    list2 = [_chunk("a"), _chunk("c")]
    merged = rrf_merge([list1, list2], k=60)
    ids = [c.id for c in merged]
    assert len(ids) == len(set(ids)) == 3


def test_chunk_without_id_ignored():
    no_id = Chunk(document_id="doc", chunk_index=0, content="x", id=None)
    merged = rrf_merge([[_chunk("a"), no_id]], k=60)
    assert [c.id for c in merged] == ["a"]
