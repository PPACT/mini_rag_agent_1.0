"""各格式解析器 —— 全部产出统一的 `ParsedDoc`（见 `blocks.py`）。

设计约束（都来自踩过的坑）：

1. **每个解析器只声明"我实际能做到什么"** —— 做不到的**不假装**。
   例如 PDF 的表格**不单独提取**：`extract_text()` 已经把表格文字带出来了，
   再单独提一遍会让同一段内容进两次检索；而 PDF 的表格结构识别质量本就不稳。
   → 这条**记进已知局限**，不写成"已支持表格"。
2. **解析器名要如实记进 `ParsedDoc.parser`** —— 路由说要用什么 ≠ 实际用了什么（协议 P-2）。
3. **非致命问题进 `warnings`**，致命问题抛错（由 `routing.py` 统一处理成显式失败）。
"""
from __future__ import annotations

from pathlib import Path

from src.document_parser.blocks import Block, ParsedDoc
from src.document_parser.tables import is_complex, render_nl, rows_to_markdown

# Excel 一个块最多放多少数据行：超过就按行分组（**重复表头**），
# 免得一整个大 sheet 变成一个超长 chunk —— 那样既切不动、也检索不准。
# 放在解析器而不是切块层，是为了让切块层只管"段落怎么切"，不必懂表格结构。
_XLSX_ROWS_PER_BLOCK = 60


def _table_block(rows: list[list[str]], caption: str | None, page: int | None = None,
                 complex_hint: bool = False, **meta) -> Block:
    """把二维表变成一个 table block（自然语言版 + 原表 + 复杂度）。

    `complex_hint`：**调用方从格式里查到的结构性事实**（如 Word 的 `gridSpan`）。
    ⚠️ 它和 `is_complex(rows)` 是**互补**的，不是重复 —— 见 `_docx_table_has_merges` 的说明。
    """
    return Block(
        kind="table",
        text=render_nl(rows, caption=caption),
        page=page,
        raw_table=rows_to_markdown(rows),
        # 「不确定时宁可判复杂」：任一判据为真即复杂（失败模式不对称，见交流区 §1.27）
        table_complex=complex_hint or is_complex(rows),
        meta=meta,
    )


# ---------------------------------------------------------------- PDF


def parse_pdf(path: str) -> ParsedDoc:
    """PDF（pdfplumber）。**逐页**产出段落块，页码 1-based。

    ⚠️ 已知局限：**表格不单独提取**（理由见模块 docstring）。PDF 里的表格文字
    仍会随 `extract_text()` 进入 `content`，所以**数值仍可被检索命中**，
    只是拿不到 `raw_table`（词法侧少一条精确值召回路径）。
    """
    import pdfplumber

    blocks: list[Block] = []
    warnings: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                blocks.append(Block(kind="paragraph", text=text, page=i))
            else:
                # 无文本层（多半是扫描件）——**记下来**，不静默跳过（2.0-3 的精神）
                warnings.append(f"第 {i} 页无文本层（可能是扫描件，需 OCR）")
        pages = len(pdf.pages)
    return ParsedDoc(blocks=blocks, parser="pdfplumber", pages=pages, warnings=warnings)


# ---------------------------------------------------------------- Word


def _iter_docx_body(doc):
    """按**文档顺序**产出段落与表格。

    ⚠️ 不能用 `doc.paragraphs` + `doc.tables` 两个集合分开取 ——
    那样会丢掉"表格在第几段之后"这个顺序信息，而顺序决定切块边界与标题归属。
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)


def _docx_table_rows(table) -> list[list[str]]:
    return [[cell.text.strip() for cell in row.cells] for row in table.rows]


# WordprocessingML 命名空间（查合并要用）
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_table_has_merges(table) -> bool:
    """查 `gridSpan`（横向合并）与 `vMerge`（纵向合并）—— **结构性事实**，不是启发式。

    ⚠️ **为什么不看文本**（2026-09-30 实测，交流区 §1.27 核对点 2）：

    一个首行"差旅标准"横跨 2 列的表，python-docx 读出来是
    `['差旅标准', '差旅标准', '备注']` —— 它把合并单元格的文本**重复**到每个被并的格里。
    于是 `is_complex()` 的两条判据（**行长不齐** / **首行有空单元格**）**都看不出 Word 的合并**，
    实测返回 `False`（漏判）。

    而底层 XML 里 `gridSpan=2` / `vMerge` 是**明确记着**的 —— 这才是口径说的
    「查 rowspan/colspan」。

    ⚠️ 漏判的代价不对称（口径原话）：**复杂误判成简单 → 静默污染知识库**；
    反过来只花钱。→ 所以这里**查到任何合并就判复杂**，不做进一步甄别。
    """
    for row in table.rows:
        for cell in row.cells:
            tcpr = cell._tc.tcPr          # noqa: SLF001 —— python-docx 没给公开 API，只能读 XML
            if tcpr is None:
                continue
            if tcpr.find(f"{_W}gridSpan") is not None or tcpr.find(f"{_W}vMerge") is not None:
                return True
    return False


def _is_heading(paragraph) -> bool:
    """判断段落是不是标题。

    中英两种都要认：`Heading 1` / `标题 1`（中文版 Word 生成的是后者）。
    """
    name = (getattr(paragraph.style, "name", "") or "").strip()
    return name.startswith("Heading") or name.startswith("标题")


def parse_docx(path: str) -> ParsedDoc:
    """Word（python-docx）。**含表格**，段落与表格按文档顺序排列。

    ⚠️ Word 没有稳定页码概念（分页由渲染决定）→ `page` 一律为 `None`。
    用 0 或 1 假装"有页码"会让引用溯源显示一个**编造**的页码，那比没有更糟。

    ⚠️ **本模块的静态局限（写在这里，不进 per-document `warnings`）**：
    `python-docx` 不暴露**目录 / 页眉 / 页脚**，故这三处内容不参与解析。
    → 它是**每一篇** docx 都成立的事实，不是"这一篇有问题"；
      塞进 `warnings` 会让那一列失去区分度（实测 21/21 个文件都 ≥1 条警告）。
    """
    import docx

    doc = docx.Document(path)
    blocks: list[Block] = []
    tables = 0

    # ⚠️ 这里**不记**"未解析目录/页眉页脚"那种警告（2026-09-30 改）：
    # 它对**每个** docx 都成立（实测语料 6/6 都有），属于**静态局限**而非"这篇有问题"。
    # 把它塞进 per-document warnings 会把那一列变成噪声 ——
    # 于是"警告数"再也分不出谁真有毛病（实测当时 21/21 个文件都 ≥1 条警告）。
    # 静态局限写在模块 docstring 里，**一次说清**：
    for item in _iter_docx_body(doc):
        if item.__class__.__name__ == "Table":
            rows = _docx_table_rows(item)
            if rows and any(any(c for c in r) for r in rows):
                blocks.append(_table_block(rows, caption=None, table_index=tables,
                                           complex_hint=_docx_table_has_merges(item)))
                tables += 1
        else:
            text = (item.text or "").strip()
            if text:
                blocks.append(Block(kind="heading" if _is_heading(item) else "paragraph",
                                    text=text))
    return ParsedDoc(blocks=blocks, parser="python-docx")


# ---------------------------------------------------------------- Excel


def parse_xlsx(path: str) -> ParsedDoc:
    """Excel（openpyxl）。**每个 sheet 一张表**；大表按行分组（重复表头）。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    blocks: list[Block] = []
    warnings: list[str] = []
    try:
        for ws in wb.worksheets:
            rows = [
                ["" if c is None else str(c).strip() for c in row]
                for row in ws.iter_rows(values_only=True)
            ]
            # 去掉整行为空的（Excel 的 used range 常拖出一堆空行）
            rows = [r for r in rows if any(c for c in r)]
            if not rows:
                warnings.append(f"sheet「{ws.title}」无内容")
                continue
            header, body = rows[0], rows[1:]
            if not body:
                blocks.append(_table_block(rows, caption=ws.title, sheet=ws.title))
                continue
            for i in range(0, len(body), _XLSX_ROWS_PER_BLOCK):
                group = [header] + body[i:i + _XLSX_ROWS_PER_BLOCK]
                part = f"{ws.title}（第 {i + 1}-{min(i + _XLSX_ROWS_PER_BLOCK, len(body))} 行）"
                blocks.append(_table_block(group, caption=part, sheet=ws.title))
    finally:
        wb.close()
    return ParsedDoc(blocks=blocks, parser="openpyxl", warnings=warnings)


# ---------------------------------------------------------------- CSV


def parse_csv(path: str) -> ParsedDoc:
    """CSV（stdlib `csv`）。整份当一个表；caption 用文件名。"""
    import csv

    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[(c or "").strip() for c in row] for row in csv.reader(f)]
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ParsedDoc(blocks=[], parser="csv", warnings=["CSV 无有效行"])
    caption = Path(path).stem
    blocks = [
        _table_block([rows[0]] + rows[1 + i:1 + i + _XLSX_ROWS_PER_BLOCK], caption=caption)
        for i in range(0, max(len(rows) - 1, 1), _XLSX_ROWS_PER_BLOCK)
    ]
    return ParsedDoc(blocks=blocks, parser="csv")


# ---------------------------------------------------------------- PPT


def parse_pptx(path: str) -> ParsedDoc:
    """PPT（python-pptx）。一张 slide 一个块；slide 内表格也提出来。"""
    import pptx

    prs = pptx.Presentation(path)
    blocks: list[Block] = []
    for i, slide in enumerate(prs.slides, 1):
        texts: list[str] = []
        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                if rows and any(any(c for c in r) for r in rows):
                    blocks.append(_table_block(rows, caption=f"第 {i} 页", slide=i))
            elif shape.has_text_frame:
                t = (shape.text_frame.text or "").strip()
                if t:
                    texts.append(t)
        if texts:
            blocks.append(Block(kind="paragraph", text="\n".join(texts),
                                meta={"slide": i}))
    return ParsedDoc(blocks=blocks, parser="python-pptx")


# ---------------------------------------------------------------- 纯文本 / Markdown / 代码


def _read_text(path: str) -> tuple[str, str]:
    """读文本文件，**顺带做编码探测**（CLN-013）。

    返回 (文本, 检测到的编码)。解码失败时抛错 —— 由 routing 变成显式失败，
    **不用 errors="ignore" 静默丢字符**（那会悄悄吃掉正文）。
    """
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is None:
            raise ValueError("无法确定文件编码（UTF-8 解码失败，charset-normalizer 也判不出）")
        return str(best), str(best.encoding)


def parse_plain(path: str) -> ParsedDoc:
    """Markdown / 纯文本 / **代码**。按空行分段；认得 `#` 标题与 ``` 代码围栏。

    ⚠️ **代码文件整块产出**（`2.0-24` 前半）：
    以前 `.py` 走的是"按空行分段"→ 实测 `23_edge_collector.py` 被切成 **20 块**，
    而 `chunking` 的「**代码整块**」策略只对 `kind == "code"` 生效 →
    **那条策略对代码文件从未生效**。现在 `.py` 直接产**一个 `code` 块**。

    ⚠️ **本轮不做 tree-sitter 按函数/类切**（那是 `2.0-24` 后半）——
    **留位置**：整块切会产出一个超长块（`23_...py` 约 250 行），
    正是将来与"按函数/类切"对比的**基线**。
    """
    text, enc = _read_text(path)
    is_md = path.lower().endswith((".md", ".markdown"))
    blocks: list[Block] = []
    warnings: list[str] = []
    if enc != "utf-8":
        warnings.append(f"编码不是 UTF-8（实测 {enc}），已转码")

    # 代码文件 → 整块（**延迟导入**避免与 routing 循环 import）
    from src.document_parser.routing import kind_by_extension

    ext = path.lower().rsplit(".", 1)[-1]
    if kind_by_extension(ext) == "code":
        body = text.strip()
        if not body:
            return ParsedDoc(blocks=[], parser="plain", warnings=[*warnings, "代码文件为空"])
        return ParsedDoc(
            blocks=[Block(kind="code", text=body, meta={"lang": ext})],
            parser="plain",
            warnings=warnings,
        )

    buf: list[str] = []
    in_code = False
    code_buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append(Block(kind="paragraph", text="\n".join(buf).strip()))
            buf.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if is_md and stripped.startswith("```"):
            if in_code:
                blocks.append(Block(kind="code", text="\n".join(code_buf).strip()))
                code_buf.clear()
            else:
                flush()
            in_code = not in_code
            continue
        if in_code:
            code_buf.append(line)
        elif is_md and stripped.startswith("#"):
            flush()
            blocks.append(Block(kind="heading", text=stripped.lstrip("# ").strip()))
        elif not stripped:
            flush()
        else:
            buf.append(line)
    flush()
    if code_buf:  # 围栏未闭合 —— 不丢内容，但要说一声
        blocks.append(Block(kind="code", text="\n".join(code_buf).strip()))
        warnings.append("代码围栏未闭合，已按普通代码块收尾")
    blocks = [b for b in blocks if b.text]
    return ParsedDoc(blocks=blocks, parser="plain", warnings=warnings)
