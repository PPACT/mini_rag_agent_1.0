"""文档加载器 —— 对外入口（2.0 起内部走三层路由）。

**新旧关系**：本模块原来是一个 `if ext == ...` 的硬链（51 行），
2.0 把解析拆成了三层（`routing.py` 文件级/解析器级 + `chunking.py` 切块级），
本模块**只保留一个兼容外壳**。

⚠️ **新代码请直接用 `load_document()`**：它返回 `ParsedDoc`，带**页码**与**原表**。
`load_text()` 只返回拼接后的纯文本 —— 它拿不到 `page` / `raw_table`，
留着是为了不破坏存量调用方（`src/tasks/document_task.py`、`eval/ingest.py`、
`eval/run_chunk_check.py`）。
"""
from __future__ import annotations

from src.document_parser.blocks import ParsedDoc
from src.document_parser.routing import parse_document


def load_document(file_path: str) -> ParsedDoc:
    """解析文档，返回**完整产物**（块 + 页码 + 原表 + 实际用的解析器）。

    失败时抛 `DocumentParseError`（带文件、路由、每个解析器各自的原因）——
    不是以前那句"不支持的文件类型: xxx"。
    """
    return parse_document(file_path)


def load_text(file_path: str) -> str:
    """按扩展名加载文档**纯文本**（兼容接口）。

    ⚠️ 表格块拼进来的是**自然语言版** —— 所以 2.0 起**表格不再被丢掉**
    （旧实现只读 `doc.paragraphs`，表格整块消失）。
    """
    return parse_document(file_path).text
