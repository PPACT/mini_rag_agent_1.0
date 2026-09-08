"""semantic_splitter 单元测试。"""
from src.document_parser.semantic_splitter import split_text


def test_empty():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_small_text_single_chunk():
    chunks = split_text("这是一段简短文字。", chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert "简短文字" in chunks[0]


def test_long_paragraph_splits_by_sentence():
    # 单段落无换行，超长时按句子切，每块不超过 chunk_size
    chunks = split_text("句子一。" * 50, chunk_size=20, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 20


def test_overlap():
    text = "第一段内容\n\n第二段内容"
    chunks = split_text(text, chunk_size=8, overlap=2)
    assert len(chunks) == 2
    # 第二个 chunk 以前一个 chunk 的尾部开头（重叠）
    assert chunks[1].startswith(chunks[0][-2:])
