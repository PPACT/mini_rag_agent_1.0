"""三层路由的第 1、2 层：**文件级** → **解析器级**（第 3 层切块见 `chunking.py`）。

设计依据：`docs/local/参考资料/RAG 数据清洗、多格式录入与切块路由方案规划表.md` §2.1 / §2.2。

⭐ **两条贯穿本模块的原则**

1. **只登记真正实现的解析器**。
   规划表 §2.2 的"首选"里列了 PyMuPDF / PaddleOCR / trafilatura / tree-sitter —— 本项目
   **都没装**。把它们写进链子，就会让路由"看起来有兜底、其实没有"，
   而失败时链条会一路走空、最后报一个看不懂的错。**没有比"假装有"更贵的了。**
2. **每一层都要有兜底，且兜底必须留下痕迹**。
   后缀撒谎 → MIME → magic number → 纯文本；
   首选解析器失败 → 链上的下一个 → 全失败则**显式抛错**（带每个解析器各自的失败原因）。
"""
from __future__ import annotations

import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.document_parser.blocks import ParsedDoc
from src.document_parser.parsers import (
    parse_csv,
    parse_docx,
    parse_pdf,
    parse_plain,
    parse_pptx,
    parse_xlsx,
)
from src.document_parser.table_recovery import recover_tables


# ---------------------------------------------------------------- 异常（2.0-3）


class DocumentParseError(Exception):
    """解析失败的统一异常。

    ⚠️ **必须带够信息让人能定位** —— 这正是 2.0-3「不再静默」的要求：
    以前只有一句 `ValueError: 不支持的文件类型: doc`，既没说哪个文件、
    也没说为什么、更没说试过什么。
    """

    def __init__(self, path: str, reason: str, detail: dict | None = None) -> None:
        self.path = path
        self.reason = reason
        self.detail = detail or {}
        super().__init__(f"{Path(path).name}：{reason}")


class UnsupportedFormatError(DocumentParseError):
    """格式已知但本轮没实现（**明确拒收，不是没认出来**）。"""


class ParseFailedError(DocumentParseError):
    """格式认得、但所有解析器都失败了。"""


# ---------------------------------------------------------------- 解析器注册表

# 解析器名 → 实现。名字与规划表 §2.2 的「首选/备选」列对齐，便于日后补备选。
_PARSER_FUNCS: dict[str, Callable[[str], ParsedDoc]] = {
    "pdfplumber": parse_pdf,
    "python-docx": parse_docx,
    "openpyxl": parse_xlsx,
    "python-pptx": parse_pptx,
    "csv": parse_csv,
    "plain": parse_plain,
}

# 类型 → 解析器链（首选 → 备选 → 降级），**全部是已实现的**。
_CHAINS: dict[str, tuple[str, ...]] = {
    "pdf": ("pdfplumber",),
    "docx": ("python-docx",),
    "xlsx": ("openpyxl",),
    "pptx": ("python-pptx",),
    "csv": ("csv",),
    "markdown": ("plain",),
    "text": ("plain",),
    "code": ("plain",),
}

# 后缀 → 类型（规划表 §2.1「后缀」行）
_EXT_KIND: dict[str, str] = {
    "pdf": "pdf", "docx": "docx", "xlsx": "xlsx", "xlsm": "xlsx", "xls": "xlsx",
    "csv": "csv", "pptx": "pptx", "md": "markdown", "markdown": "markdown",
    "txt": "text", "log": "text",
    "py": "code", "js": "code", "ts": "code", "java": "code", "go": "code",
    "rs": "code", "c": "code", "h": "code", "cpp": "code", "cs": "code",
    "rb": "code", "php": "code", "sh": "code", "sql": "code", "yaml": "code",
    "yml": "code", "toml": "code", "ini": "code", "json": "code", "xml": "code",
}

# ⚠️ 后缀**认得**、但本轮**不实现** → 显式拒收并给出原因。
#    与"认不出来"是两回事：前者说明我们评估过、明确不做；后者才是意外。
_KNOWN_UNSUPPORTED: dict[str, str] = {
    "doc": "旧版二进制 .doc —— python-docx 读不了，需先转成 .docx（LibreOffice 可转）",
    "ppt": "旧版二进制 .ppt —— python-pptx 读不了，需先转成 .pptx",
    "html": "HTML —— 需 trafilatura / beautifulsoup4 提取正文（本项目未接入）",
    "htm": "HTML —— 同 .html",
    "rtf": "RTF —— 无解析器",
    "odt": "ODT —— 无解析器（可先用 LibreOffice 转 .docx）",
    "epub": "EPUB —— 无解析器",
    "eml": "邮件 —— 无解析器",
    "msg": "Outlook 邮件 —— 无解析器",
    "zip": "压缩包 —— 需解压后递归路由（本轮未实现）",
}

# magic number → 类型（规划表 §2.1「内容」行）—— 用于**后缀缺失或撒谎**时
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "_zip"),          # Office 三兄弟都是 zip，需再看内部结构
    (b"{\\rtf", "rtf"),
    (b"\xd0\xcf\x11\xe0", "doc"),     # 旧版 Office 复合文档（OLE2）
)

# zip 内部目录名 → 类型
_ZIP_MARKERS: tuple[tuple[str, str], ...] = (
    ("word/", "docx"), ("xl/", "xlsx"), ("ppt/", "pptx"),
)


def supported_extensions() -> frozenset[str]:
    """**本模块是"支持哪些格式"的唯一事实源** —— 供上传接口的校验直接引用。

    ⚠️ 为什么要有这个函数：`upload_api` 原来自己维护了一份 `_ALLOWED`，里面赫然写着
    `doc` / `ppt` —— 而这两个**从来就读不了**（python-docx/pptx 都读不了旧版二进制格式）。
    于是上传时放行、进 worker 才失败。**两份清单必然漂移**，所以让它们只有一份。
    """
    return frozenset(e for e, k in _EXT_KIND.items() if k in _CHAINS)


@dataclass(frozen=True)
class RoutePlan:
    """文件级 + 解析器级的决策结果（可打印、可断言、可审计）。"""

    kind: str
    chain: tuple[str, ...]
    confidence: str                      # high / medium / low
    signals: tuple[str, ...]             # 命中了哪些信号（后缀 / MIME / magic）
    note: str = ""

    def describe(self) -> str:
        return (f"kind={self.kind} chain={'→'.join(self.chain) or '(空)'} "
                f"confidence={self.confidence} signals={'+'.join(self.signals)}")


def _kind_by_magic(path: str) -> str | None:
    """按文件头字节判断类型（后缀不可信时用）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    if head.startswith((b"<html", b"<!DOCTYPE html", b"<!doctype html")):
        return "html"
    return None


def _kind_by_zip(path: str) -> str | None:
    """zip 容器再看内部目录，区分 docx / xlsx / pptx。"""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()[:200]
    except (zipfile.BadZipFile, OSError):
        return None
    for marker, kind in _ZIP_MARKERS:
        if any(n.startswith(marker) for n in names):
            return kind
    return None


def route_file(path: str) -> RoutePlan:
    """文件级路由：**后缀 + magic number 相互印证 → MIME → 纯文本兜底**。

    ⭐ **优先级的关键取舍**：`magic` 读的是**文件内容本身**，不会撒谎；
    后缀只是文件名。所以**两者冲突时以 magic 为准**，并把冲突记进 `signals`
    （协议 P-2：要能核实"实际是靠哪条信号判出来的"）。

    ⚠️ 但**policy 优先于 magic**：后缀落在"认得但不做"表里（如 `.doc`）时**直接拒收**，
    不让 magic 把它救回来 —— 那是**明确的产品决定**，不是识别失败。
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    signals: list[str] = []
    note = ""

    # ① 后缀：拿到候选类型；落在"认得但不做"表里则**直接拒收**
    if ext in _KNOWN_UNSUPPORTED:
        raise UnsupportedFormatError(path, f"暂不支持该格式：{_KNOWN_UNSUPPORTED[ext]}",
                                     {"ext": ext, "signal": "后缀"})
    ext_kind = _EXT_KIND.get(ext)

    # ② magic number（内容硬证据）—— 后缀缺失、或后缀在撒谎时都靠它
    magic_kind = _kind_by_magic(path)
    if magic_kind == "_zip":
        magic_kind = _kind_by_zip(path)
    if magic_kind in _KNOWN_UNSUPPORTED:
        raise UnsupportedFormatError(
            path, f"暂不支持该格式：{_KNOWN_UNSUPPORTED[magic_kind]}",
            {"kind": magic_kind, "signal": "magic"})

    # ③ 定夺
    if magic_kind and magic_kind != ext_kind:
        kind, confidence = magic_kind, "high"
        signals.append(f"magic.{magic_kind}")
        signals.append(f"冲突.后缀={ext or '无'}")
        note = (f"后缀与内容不符（后缀={ext or '无'}，实际内容={magic_kind}）—— "
                f"**已按内容解析**；若文件确实损坏，请重新导出")
    elif ext_kind:
        kind, confidence = ext_kind, "high"
        signals.append(f"后缀.{ext}")
    else:
        # ④ MIME（置信度「中」）
        guessed, _ = mimetypes.guess_type(p.name)
        sub = guessed.split("/")[-1].lower() if guessed else ""
        kind = _EXT_KIND.get(sub)
        if kind:
            signals.append(f"MIME.{guessed}")
            confidence = "medium"
        else:
            # ⑤ 兜底：当纯文本读（规划表 §2.1 的「纯文本兜底」）
            kind, confidence = "text", "low"
            signals.append("兜底.纯文本")

    if kind not in _CHAINS:                 # 理论上到不了（两张表已对齐）
        kind, confidence = "text", "low"
        signals.append("兜底.纯文本")

    chain = _CHAINS[kind]
    if len(chain) == 1 and not note:
        note = "该类型只有一个解析器（规划表的备选均未接入）"
    return RoutePlan(kind=kind, chain=chain, confidence=confidence,
                     signals=tuple(signals), note=note)


def parse_document(path: str) -> ParsedDoc:
    """路由 → 依次尝试链上的解析器 → 全失败则**显式抛错**。

    与旧 `load_text()` 的关键区别：**失败时说的是"哪个文件、用的什么解析器、各自为什么失败"**，
    而不是一句"不支持的文件类型"。
    """
    plan = route_file(path)
    attempts: list[tuple[str, str]] = []

    for name in plan.chain:
        fn = _PARSER_FUNCS.get(name)
        if fn is None:                       # 理论上到不了（_CHAINS 已校验）
            attempts.append((name, "解析器未登记（路由表配置错误）"))
            continue
        try:
            doc = fn(path)
        except Exception as e:  # noqa: BLE001 —— 任何解析器异常都该走兜底，而不是中断整条管道
            attempts.append((name, f"{type(e).__name__}: {e}"))
            continue
        if not doc.blocks:
            attempts.append((name, "解析成功但没有任何内容（空文档 / 内容全被过滤）"))
            continue
        # 2.0-1 补缺口：把"被降级成普通段落"的 Markdown 表格**重建**成 table 块
        # （md→docx 的转换工具会把表格转成 `| a | b |` 这种普通文本；
        #  不重建则那些文件的表格块数为 0 → 表格数值命中率的**分母是假的**）
        # ⚠️ 放在这里而不是调用方：让**所有**解析入口自动受益，不能靠"记得调"
        doc, _recovery = recover_tables(doc)
        if plan.note:
            doc.warnings.insert(0, plan.note)
        return doc

    raise ParseFailedError(
        path,
        "所有解析器都失败了（该格式的解析器链已走完）",
        {"route": plan.describe(), "attempts": attempts},
    )


def failure_report(err: DocumentParseError) -> str:
    """把失败渲染成人能读的一行（供批量灌库的**失败文件清单**用，2.0-3）。"""
    lines = [f"❌ {err.path}", f"   原因：{err.reason}"]
    if route := err.detail.get("route"):
        lines.append(f"   路由：{route}")
    for name, why in err.detail.get("attempts", []):
        lines.append(f"   解析器 {name}：{why}")
    return "\n".join(lines)
