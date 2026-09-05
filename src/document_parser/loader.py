"""文档加载器：PDF / Word / PPT -> 纯文本（免费方案，可插拔）。"""
from __future__ import annotations


def load_text(file_path: str) -> str:
    """按扩展名加载文档文本。"""
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _load_pdf(file_path)
    if ext in ("docx", "doc"):
        return _load_docx(file_path)
    if ext in ("pptx", "ppt"):
        return _load_pptx(file_path)
    raise ValueError(f"不支持的文件类型: {ext}")


def _load_pdf(path: str) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _load_docx(path: str) -> str:
    import docx

    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _load_pptx(path: str) -> str:
    import pptx

    prs = pptx.Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n".join(parts)
