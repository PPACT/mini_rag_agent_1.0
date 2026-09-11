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
    para = "这是一段测试文本用来验证重叠逻辑是否正确实现。"
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = split_text(text, chunk_size=25, overlap=5)
    assert len(chunks) >= 2
    # 重叠按比例封顶后依然生效
    assert chunks[1].text.startswith(chunks[0].text[-5:])


def test_heading_title():
    chunks = split_text("考勤制度\n员工每天上午9点上班，中午12点休息。", chunk_size=100, overlap=0)
    assert any(c.title == "考勤制度" for c in chunks)


def test_offsets_in_bounds():
    text = "第一段。\n\n第二段。\n\n第三段。"
    chunks = split_text(text, chunk_size=6, overlap=0)
    for c in chunks:
        assert 0 <= c.start <= c.end <= len(text)


# ---- 回归：修复"标题单独成块"与"短块重叠过度" ----

def test_consecutive_headings_not_tiny_chunks():
    """连续标题不应产出只有标题的极小废块。"""
    text = "# 新员工入职指引\n## 入职材料\n新员工报到当天需携带身份证原件和学历证书。"
    chunks = split_text(text, chunk_size=512, overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0].text) > 20
    assert "学历证书" in chunks[0].text


def test_no_tiny_chunks_on_structured_doc():
    """结构化文档（多标题）不应产出过小的块。"""
    text = (
        "# 考勤制度\n## 工作时间\n公司标准工作时间为上午9点到下午6点。\n"
        "## 打卡要求\n员工须每日打卡两次，忘记打卡可申请补卡。\n"
        "## 迟到处理\n迟到30分钟以内记警告一次，超过按半天事假处理。"
    )
    chunks = split_text(text, chunk_size=512, overlap=64)
    assert all(len(c.text) >= 20 for c in chunks), [len(c.text) for c in chunks]


def test_overlap_capped_on_short_chunks():
    """短块的重叠不应超过其长度的 25%（避免近半重复）。"""
    para = "短段落内容一二三。"  # 9 字符
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = split_text(text, chunk_size=10, overlap=8)
    # 重叠被按比例封顶（9*0.25=2），而非直接用 8
    assert chunks[1].text.startswith(chunks[0].text[-2:])
    assert len(chunks[1].text) < len(chunks[0].text) + 8
