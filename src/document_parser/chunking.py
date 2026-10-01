"""切块级路由（2.0-2 的第 3 层）—— 按**块类型**决定怎么切。

规划表 §2.3 列了 10 种策略；本轮实现 3 条，其余**明确不做**（不写"看起来支持"的分支）。

| 块类型 | 策略 | 为什么 |
|---|---|---|
| `table` | **整块，不切** | 表格从中间切断 = 行列头与数值分离：`住宿上限=600` 变成裸的 `600` —— **数值还在、语义没了** |
| `code` | **整块，不切** | 同源：被切断的函数/类既检索不准、人也没法用 |
| 段落 / 标题 | 复用现有 `split_text` | 它已做了标题累积 + 超长段按句切 + 短块合并，**没理由重写** |

⚠️ **两个必须守住的细节**（都不是我发明的，是读代码读出来的）：

1. **不能逐块调用 `split_text`**。它靠**跨段维护"当前章节"**：标题被累积进当前块，
   同时记为该块的 `title`。逐块调用 → 标题与它后面的段落被割开 → **章节归属全丢**。
   → 所以按"**段**"切：连续文本块合为一段，段内保持原有行为。
2. **页边界也断段**。PDF 是"一页一块"，若把多页并成一段，`page` 只能取首块 →
   **后面几页的内容会被标上第一页的页码** —— 引用溯源于是显示一个**看起来可信、其实是编造**的页码。
   断段后 `page` 是精确的。

⚠️ **偏移语义的变化**：`start`/`end` 现在是**相对于所属段**，不再是"相对于整份文档的拼接文本"。
（后者在块级切块下已无意义。）当前**没有任何消费方**（已 `grep` 确认只有两处赋值、无处读取），
将来要做引用高亮时，需连同块信息一起用。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.document_parser.blocks import Block, ParsedDoc
from src.document_parser.semantic_splitter import split_text
from src.document_parser.tables import with_caption

# 整块不切的类型
_ATOMIC = ("table", "code")


@dataclass
class ChunkCandidate:
    """切块层的产物：文本 + 位置 + **来自解析层的元数据**。

    与 `TextChunk` 的分工：`TextChunk` 只管"文本怎么切"（`semantic_splitter` 的职责）；
    这里多带的 `page` / `raw_table` / `table_complex` 是**解析**给的，不是切块算出来的 ——
    所以不塞进 `TextChunk`，免得那一层无端背上表格的概念。
    """

    text: str
    start: int
    end: int
    title: str | None = None
    page: int | None = None
    raw_table: str | None = None
    table_complex: bool = False
    kind: str = "paragraph"


def _segments(blocks: list[Block]) -> list[tuple[str, list[Block]]]:
    """把块切成"段"：连续的文本块为一段；表格/代码各成一段；**页边界也断段**。

    返回 [(段类型, 段内块)]，段类型取 `"text"` 或原子块的 kind。
    """
    out: list[tuple[str, list[Block]]] = []
    cur: list[Block] = []

    def flush() -> None:
        if cur:
            out.append(("text", list(cur)))
            cur.clear()

    for b in blocks:
        if b.kind in _ATOMIC:
            flush()
            out.append((b.kind, [b]))
            continue
        if cur and cur[-1].page != b.page:      # ① 页边界断段（保证 page 精确）
            flush()
        cur.append(b)
    flush()
    return out


def _merge_heading_captions(segs: list[tuple[str, list[Block]]]) -> list[tuple[str, list[Block]]]:
    """`2.0-46`：把「**只含 heading 的 text 段** + 紧随的 table 段」合成一组，heading 作 caption。

    ⚠️ **为什么落点在这里、不在解析层**（文档侧 §1.37②，我认同）：
    heading 同时是**章节标记** —— `split_blocks` 靠它维护 `heading` 变量、喂给后续块的 `title`。
    在解析层把它"吞成 caption"会**丢掉层级**。这里合并的只是**段的组合方式**：
    IR 不变，且合并后 `heading` 变量**照常更新**（见 `split_blocks` 里的 `heading = cap.text`）。

    **判据（结构性，不是猜）**：

        heading 紧邻（块序号差 1）且下一块是 table  →  合并
        中间夹了任何块（段落 / 另一个 heading）      →  不合并

    ⚠️ **宁可漏判** —— 漏判只是多一个碎块；**误判是把章节标题错当成 caption**（污染表格语义）。
    """
    out: list[tuple[str, list[Block]]] = []
    i = 0
    while i < len(segs):
        kind, group = segs[i]
        if (kind == "text" and len(group) == 1 and group[0].kind == "heading"
                and i + 1 < len(segs) and segs[i + 1][0] == "table"):
            out.append(("table_captioned", [group[0], *segs[i + 1][1]]))
            i += 2
            continue
        out.append((kind, group))
        i += 1
    return out


def split_blocks(parsed: ParsedDoc, chunk_size: int = 512,
                 overlap: int = 64) -> list[ChunkCandidate]:
    """按块类型切块（本文件是它的说明书，见模块 docstring）。"""
    out: list[ChunkCandidate] = []
    heading: str | None = None      # 跨段跟踪的"当前章节"

    for kind, group in _merge_heading_captions(_segments(parsed.blocks)):
        if kind == "table_captioned":
            # `2.0-46`：紧邻 heading 的表格 → heading 既作 caption、**也仍是章节标记**
            cap, tb = group[0], group[1]
            out.append(ChunkCandidate(
                text=with_caption(tb.text, cap.text),
                start=0, end=len(tb.text), title=cap.text,
                page=tb.page, raw_table=tb.raw_table,
                table_complex=tb.table_complex, kind="table",
            ))
            heading = cap.text      # ⚠️ **层级不丢**：被吞的 heading 仍喂给后续块
            continue

        if kind != "text":
            b = group[0]
            out.append(ChunkCandidate(
                text=b.text, start=0, end=len(b.text), title=heading,
                page=b.page, raw_table=b.raw_table,
                table_complex=b.table_complex, kind=kind,
            ))
            continue

        text = "\n".join(b.text for b in group)
        page = group[0].page
        for tc in split_text(text, chunk_size, overlap):
            out.append(ChunkCandidate(
                text=tc.text, start=tc.start, end=tc.end,
                # 段内自己认出的标题优先；没有则用**跨段跟踪的章节**
                title=tc.title or heading,
                page=page, kind="paragraph",
            ))
        # 本段里最后出现的标题，成为后续段（如表格之后的段落）的章节
        for b in group:
            if b.kind == "heading":
                heading = b.text

    return out
