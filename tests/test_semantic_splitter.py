"""semantic_splitter 单元测试。"""
from src.document_parser.semantic_splitter import split_text


def test_empty():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []


def test_small_text_single_chunk():
    chunks = split_text("这是一段简短文字。", chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert "简短文字" in chunks[0].text


def test_long_paragraph_splits_by_sentence():
    chunks = split_text("句子一。" * 50, chunk_size=20, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 20


def test_overlap():
    chunks = split_text("第一段。\n\n第二段。\n\n第三段。", chunk_size=6, overlap=2)
    assert len(chunks) >= 2
    # 第二个 chunk 的文本以前一个 chunk 的尾部开头（重叠）
    assert chunks[1].text.startswith(chunks[0].text[-2:])


def test_heading_title():
    chunks = split_text("考勤制度\n员工每天上午9点上班，中午12点休息。", chunk_size=100, overlap=0)
    assert any(c.title == "考勤制度" for c in chunks)


def test_offsets_in_bounds():
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = split_text(text, chunk_size=6, overlap=0)
    for c in chunks:
        assert 0 <= c.start <= c.end <= len(text)
