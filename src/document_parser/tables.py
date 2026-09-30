"""表格处理：原表 → **自然语言版** + **原表 Markdown** + 复杂度判定 + 数值保真校验。

口径（交流区 §1.24 Q5，用户已裁决）：

| 产出 | 去向 | 向量 | 词法 |
|---|---|---|---|
| **自然语言版** | `chunks.content` | ✅ | ✅ |
| **原表 Markdown** | `chunks.raw_table` | ❌ | ✅ |

**为什么向量不吃原表**：`| Q3 | 5000 |` 这种裸表的语义**依赖行列头**，单 chunk 的向量抓不住；
且吃它会**挤占向量 top-k 席位**。而 RRF 按 `chunk_id` 去重 ——
同一 chunk 走两路命中**只算一次**，不占两个席位。

⭐ **本模块的核心设计**：自然语言渲染时**单元格值逐字保留**（不换算、不四舍五入、不省略单位）。
于是「原表里的每个数值都能在自然语言版里找到」这条验收**几乎是自动成立的** ——
而 `missing_numbers()` 就是守它的闸：**每次改动都能自动复验**，不靠人眼。

⚠️ **一处实测修正（2026-09-30）——别把 `raw_table` 进词法的作用说大了**：
因为上面那条保真成立，`tokenize(content + raw_table)` 与 `tokenize(content)`
的 **token 集合完全相同**（实测 `raw_table` **独有的 token 数 = 0**）。
所以「原表进词法 = 多一条精确命中路径」这个说法**是错的**，真实作用是：

| 作用 | 是真是假 |
|---|---|
| **词频提升**（ts_rank 计次 → 数值类查询下表格块排名略靠前） | ✅ 真实 |
| **保险**（将来自然语言版若变得有损，原表还在索引里） | ✅ 真实 |
| "新增可搜内容 / 精确命中路径" | ❌ **实测不成立** |
| **给人在引用里核对原表**（见 `Source.raw_table`） | ✅ 真实，且是"保留原表"的主要价值 |

→ 结论：**保留这个设计**（无害、便宜、且是已裁决口径），但**别依赖一条不存在的召回路径**。
"""
from __future__ import annotations

import re

# 数值 token：带可选符号、允许**千分位逗号在匹配内部**、可选小数。
#
# ⚠️ 两处都是踩过才写对的：
# - **逗号必须在正则里**：若写成 `\d+(\.\d+)?`，`1,000` 会被拆成 `1` 与 `000` 两个 token
#   —— 归一化再怎么写都救不回来（逗号在匹配之外）。
# - **符号要纳入**：`-300` 与 `300` 若都归一成 `300`，那么"数值写错正负"这种真丢失
#   会被这道闸放过去。
_NUM = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def _norm(num: str) -> str:
    """数值归一化：`1,000` 与 `1000` 视为同一个数（格式化差异不算丢失）。"""
    return num.replace(",", "")


def number_tokens(text: str) -> list[str]:
    """抽出文本里的数值 token（已归一化、已去重，保持出现顺序）。

    ⚠️ 去重是刻意的：`missing_numbers` 的返回值要给人看，
    同一个数缺 5 次报 5 行只是噪声。
    """
    seen: list[str] = []
    for raw in _NUM.findall(text or ""):
        n = _norm(raw)
        if n not in seen:
            seen.append(n)
    return seen


def missing_numbers(raw_table: str, nl_text: str) -> list[str]:
    """**2.0-1 的验收闸**：原表里的每个数值，都必须能在「自然语言版」里找到。

    返回**缺失的**数值（空列表 = 通过）。机械可查，不靠人眼。

    ⚠️ 方向是**单向**的：只查「原表 ⊄ 自然语言版」。
    自然语言版里多出来的数字（比如补了"共 3 项"）**不算失败** ——
    验收要防的是**丢信息**，不是防**多解释**。
    """
    have = set(number_tokens(nl_text))
    return [n for n in number_tokens(raw_table) if n not in have]


def is_complex(rows: list[list[str]]) -> bool:
    """判断是否「复杂表」—— 复杂表的**原表也要进向量**（2.0-1 口径）。

    判据是**启发式**（合并单元格在各库里的表现不一致，无法百分百可靠）：

    - **列数不齐**：同一表里行长不同 → 多半有合并单元格
    - **表头有空单元格**：Excel 的合并单元格只有锚点有值 → 表头出现空格

    ⚠️ **抓不到 Word 的合并**（2026-09-30 实测）：python-docx 会把合并单元格的文本
    **重复**到每个被并的格里（`差旅标准 | 差旅标准 | 备注`），所以上面两条**都不成立**。
    → Word 那边由 `parsers._docx_table_has_merges()` 查 XML 的 `gridSpan` / `vMerge` 补上
    （即口径说的「查 rowspan/colspan」）。**两个判据互补，缺一不可。**

    ⚠️ 它是"要不要多进一路向量"的**保守开关**，不是正确性判据 ——
    真正的正确性由 `missing_numbers()` 守。误判成 complex 只是多占一点向量空间。
    """
    if not rows:
        return False
    if len({len(r) for r in rows}) > 1:
        return True
    return any(not str(c).strip() for c in rows[0])


def rows_to_markdown(rows: list[list[str]], header: bool = True) -> str:
    """把二维表渲染成 Markdown 表（`raw_table` 就存这个）。

    ⚠️ 单元格里的 `|` 需转义，否则整张表的列会错位 —— 而错位之后
    **词法检索仍能命中数值**，所以这种错误很晚才会被发现。
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    out: list[str] = []

    def _line(cells: list[str]) -> str:
        padded = [str(c).replace("|", "\\|").strip() for c in cells] + [""] * (width - len(cells))
        return "| " + " | ".join(padded) + " |"

    head = rows[0] if header else [f"列{i + 1}" for i in range(width)]
    out.append(_line(head))
    out.append("| " + " | ".join(["---"] * width) + " |")
    for r in rows[1:]:
        out.append(_line(r))
    return "\n".join(out)


def render_nl(rows: list[list[str]], caption: str | None = None) -> str:
    """把表格渲染成**自然语言版**（`chunks.content` 里表格那部分）。

    格式：`表「<caption>」：<列1>=<值>，<列2>=<值>；<下一行>…`

    设计要点：
    - **每行的列名与值都写出**（不靠"同上"省略）→ 语义完整、向量能匹配；
    - **值逐字保留**（不换算、不加单位、不四舍五入）→ 数值保真**天然成立**；
    - 表头缺失时用「列N」占位 —— 宁可承认"不知道列名"，也不编一个。
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    header = [str(c).strip() for c in rows[0]]
    # 表头若为空单元格（Excel 合并所致），补成"列N"，避免渲染出 `=` 这种空壳
    header = [h if h else f"列{i + 1}" for i, h in enumerate(header)]
    if len(header) < width:
        header += [f"列{i + 1}" for i in range(len(header), width)]

    prefix = f"表「{caption}」：" if caption else "表："
    body = rows[1:] if len(rows) > 1 else []
    if not body:
        return prefix + "；".join(f"{header[i]}={header[i]}" for i in range(width))

    lines: list[str] = []
    for r in body:
        cells = [str(c).strip() for c in r] + [""] * (width - len(r))
        lines.append("；".join(f"{header[i]}={cells[i]}" for i in range(width)))
    return prefix + "；".join(lines)
