"""`2.0-48` **先量** —— Word 里「格式上是标题、却没被认成 `heading`」的段落有多少。

## 为什么要先量（`改进计划.md` 阶段 0-f 候选 c / `交流区 §1.37`）

`2.0-46` 实测极短块 **9 → 7**（`bca4156`），**漏判 7/9**。
根因：`三、成本与费用` 这类段落**没被 Word 样式标成 `Heading`** →
解析层判成 `paragraph` → **切块层的判据够不着**。

三种判据分层（文档侧口径）：

| 判据 | 性质 | 该放哪 |
|---|---|---|
| Word **Heading 样式** | ✅ 事实 | 解析层（**已在用**，见 `parsers._is_heading`） |
| Word **加粗 / 字号** | 🟡 格式事实 | ⭐ 解析层（**← 本脚本就是要量它有多少对象**） |
| **文本正则** `^[一二三]+、` | ❌ 猜语义 | ⛔ 哪都不该放（**本脚本不测**，测了就是造指标） |

## 纪律

- **只量不改**：**不预设阈值、不给建议、不下结论** —— 只出数字与明细，判据由文档侧裁。
- **同一事实只留一个来源**：heading 判定**直接复用** `parsers._is_heading`，
  段落遍历**直接复用** `parsers._iter_docx_body` —— 不在这里重写一份（重写必然漂移）。
- **极短块阈值**复用 `eval/l0_report.py` 的 `TINY_CHARS`（同一口径，别各写一份）。
- 加粗/字号要**逐层解析继承**（run → 段落标记 `pPr/rPr` → 样式链 → **`docDefaults`**）——
  只看 `run.bold` 会把"用段落格式刷成粗体"的整批漏掉；
  而**漏掉 `docDefaults` 会让大部分正文的字号解析成 `None`** ——
  ⚠️ 实测踩过：`07` 的 40 段里 **32 段是 `None`**，于是"正文基准字号"只能从
  **少数几个显式带字号的段落**（恰恰就是标题）里算出 → **基准 = 拿标题当正文**，
  `D1/D2` 两档数字**整批失真**。所以这一层必须有。

## 它回答什么

1. 每个 docx：段总数 / 现判 heading 数 / **各档"格式像标题"的候选数**
2. **`2.0-46` 残留的极短块**逐条列出，并标注**格式判据能否捞回它**
3. 全量候选明细（写 `logs/probe_docx_heading.json`）

用法：
    python eval/probe_docx_heading.py
    python eval/probe_docx_heading.py --dir docs/corpus --json logs/probe_docx_heading.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.document_parser.chunking import split_blocks  # noqa: E402
from src.document_parser.parsers import _is_heading, _iter_docx_body  # noqa: E402
from src.document_parser.routing import parse_document  # noqa: E402

# ⚠️ 与 `eval/l0_report.py` **同一口径** —— 两处各写一份必然漂移（dev-agent 纪律）
TINY_CHARS = 50
OUT_DEFAULT = ROOT / "logs" / "probe_docx_heading.json"
# 仅用于**分档报数**的展示宽度，**不是判据**
SHORT_CHARS = 40


# ---------------------------------------------------------------- 格式解析


def _el_bool(el) -> bool | None:
    """`<w:b/>` → True；`<w:b w:val="0"/>` → False；元素不存在 → None（= 继承）。"""
    if el is None:
        return None
    v = el.get(qn("w:val"))
    if v is None:
        return True
    return str(v).lower() not in ("0", "false", "off")


def _rpr_bold(rpr) -> bool | None:
    return _el_bool(rpr.find(qn("w:b"))) if rpr is not None else None


def _rpr_size_pt(rpr) -> float | None:
    """`<w:sz w:val="32"/>` = 16pt（半磅）。"""
    if rpr is None:
        return None
    sz = rpr.find(qn("w:sz"))
    if sz is None:
        return None
    try:
        return int(sz.get(qn("w:val"))) / 2.0
    except (TypeError, ValueError):
        return None


def _style_chain(style, getter):
    """沿 `base_style` 链找第一个非 None —— 样式继承有深度，只看一层会漏。"""
    seen: set[int] = set()
    st = style
    while st is not None and id(st) not in seen:
        seen.add(id(st))
        val = getter(st)
        if val is not None:
            return val
        st = st.base_style
    return None


def _style_bold(para) -> bool | None:
    return _style_chain(para.style, lambda s: getattr(s.font, "bold", None))


def _style_size_pt(para) -> float | None:
    return _style_chain(
        para.style, lambda s: s.font.size.pt if s.font.size is not None else None
    )


def _doc_default_marks(doc) -> tuple[bool | None, float | None]:
    """**继承链的最后一环**：`w:styles/w:docDefaults/w:rPrDefault/w:rPr`。

    ⚠️ python-docx **不暴露** `docDefaults` —— 而绝大多数"没显式设字号"的正文，
    真实字号**只**能从这里读到。不读它，正文全变 `None`（见模块 docstring 的实测）。
    """
    styles_el = getattr(doc.styles, "element", None)
    if styles_el is None:
        return None, None
    dd = styles_el.find(qn("w:docDefaults"))
    if dd is None:
        return None, None
    rprdef = dd.find(qn("w:rPrDefault"))
    rpr = rprdef.find(qn("w:rPr")) if rprdef is not None else None
    return _rpr_bold(rpr), _rpr_size_pt(rpr)


def _para_marks(para, doc_bold: bool | None, doc_size: float | None) -> dict:
    """一个段落的**格式事实**（逐层解析继承，全部只读 XML，不猜文本）。"""
    ppr = para._p.find(qn("w:pPr"))
    ppr_rpr = ppr.find(qn("w:rPr")) if ppr is not None else None

    par_bold = _rpr_bold(ppr_rpr)          # 段落标记默认（格式刷整段加粗会落这）
    par_size = _rpr_size_pt(ppr_rpr)
    sty_bold = _style_bold(para)
    sty_size = _style_size_pt(para)

    eff_bolds: list[bool] = []
    eff_sizes: list[float] = []
    for r in para.runs:
        if not (r.text or "").strip():
            continue
        rpr = r._r.find(qn("w:rPr"))
        b = _rpr_bold(rpr)
        for fallback in (par_bold, sty_bold, doc_bold):   # 逐层回退，最后才是 docDefaults
            if b is None:
                b = fallback
        sz = _rpr_size_pt(rpr)
        for fallback in (par_size, sty_size, doc_size):
            if sz is None:
                sz = fallback
        eff_bolds.append(bool(b) if b is not None else False)
        if sz is not None:
            eff_sizes.append(sz)

    text = (para.text or "").strip()
    return {
        "text": text,
        "len": len(text),
        "style": (getattr(para.style, "name", "") or "").strip(),
        "is_heading": _is_heading(para),
        "bold_all": bool(eff_bolds) and all(eff_bolds),
        "bold_any": any(eff_bolds),
        "size_pt": max(eff_sizes) if eff_sizes else None,
        "style_bold": sty_bold,
        "par_bold": par_bold,
    }


def _median_size(paras: list[dict]) -> float | None:
    """正文基准字号 —— 取「非 heading 且非全段加粗」段落的字号中位数。

    若一个都没有（整篇都加粗这种怪文档），退回全体中位数；再没有则 None。
    """
    body = [p["size_pt"] for p in paras
            if not p["is_heading"] and not p["bold_all"] and p["size_pt"] is not None]
    pool = body or [p["size_pt"] for p in paras if p["size_pt"] is not None]
    return statistics.median(pool) if pool else None


# ---------------------------------------------------------------- 单文件


def _scan_docx(path: Path, chunk_size: int, overlap: int) -> dict:
    """一个 docx 的：段落格式事实 + `2.0-46` 残留极短块 + 各档候选计数。"""
    import docx  # 延迟导入，和 parsers.parse_docx 保持一致

    rec: dict = {"file": path.name}
    try:
        d = docx.Document(str(path))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    doc_bold, doc_size = _doc_default_marks(d)
    rec["doc_defaults"] = {"bold": doc_bold, "size_pt": doc_size}

    paras = [_para_marks(p, doc_bold, doc_size)
             for p in _iter_docx_body(d) if hasattr(p, "runs")]
    paras = [p for p in paras if p["text"]]
    # ⚠️ 自检：字号解析不到的段落数 —— 这个数一旦很大，"基准"就不可信（见模块 docstring）
    rec["n_size_unresolved"] = sum(1 for p in paras if p["size_pt"] is None)
    baseline = _median_size(paras)

    # ---- 各档候选（**只分档报数，不定判据**）----
    not_head = [p for p in paras if not p["is_heading"]]
    c_bold = [p for p in not_head if p["bold_all"]]
    c_bold_short = [p for p in c_bold if p["len"] <= SHORT_CHARS]
    c_bold_short_nopunct = [
        p for p in c_bold_short if p["text"][-1] not in "。！？；：，,.;:!?"
    ]
    c_big = [p for p in not_head
             if p["size_pt"] is not None and baseline is not None and p["size_pt"] > baseline]
    c_big_short = [p for p in c_big if p["len"] <= SHORT_CHARS]

    # ---- `2.0-46` 残留极短块：格式判据能不能捞回它 ----
    tiny: list[dict] = []
    try:
        chunks = split_blocks(parse_document(str(path)), chunk_size, overlap)
        by_text = {p["text"]: p for p in paras}
        for c in chunks:
            t = (c.text or "").strip()
            if len(t) >= TINY_CHARS:
                continue
            hit = by_text.get(t)
            tiny.append({
                "text": t,
                "len": len(t),
                "single_para": hit is not None,
                "is_heading": hit["is_heading"] if hit else None,
                "bold_all": hit["bold_all"] if hit else None,
                "size_pt": hit["size_pt"] if hit else None,
            })
    except Exception as e:  # noqa: BLE001
        rec["chunk_error"] = f"{type(e).__name__}: {e}"

    rec.update(
        n_paras=len(paras),
        n_heading=sum(1 for p in paras if p["is_heading"]),
        n_not_heading=len(not_head),
        baseline_size_pt=baseline,
        c1_bold=len(c_bold),
        c2_bold_short=len(c_bold_short),
        c3_bold_short_nopunct=len(c_bold_short_nopunct),
        d1_big=len(c_big),
        d2_big_short=len(c_big_short),
        tiny_chunks=tiny,
        cand_bold_short=[_brief(p) for p in c_bold_short],
        cand_big_short=[_brief(p) for p in c_big_short],
        all_paras=paras,
    )
    return rec


def _brief(p: dict) -> dict:
    return {k: p[k] for k in ("text", "len", "style", "bold_all", "size_pt")}


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "docs" / "corpus"))
    ap.add_argument("--json", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    s = get_settings()
    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"[X] 目录不存在：{src_dir}")
        return 1

    files = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".docx")
    if not files:
        print(f"[X] {src_dir} 下没有 .docx")
        return 1

    recs = [_scan_docx(p, s.chunk_size, s.chunk_overlap) for p in files]
    ok = [r for r in recs if not r.get("error")]

    print(f"2.0-48 探针 ｜ Word 标题格式事实 ｜ {src_dir}")
    print(f"参数：chunk_size={s.chunk_size}  overlap={s.chunk_overlap} ｜ "
          f"极短块阈值 <{TINY_CHARS} 字符 ｜ 短标题展示宽度 ≤{SHORT_CHARS} 字符（非判据）\n")

    print(f"{'文件':<34}{'段':>4}{'heading':>8}{'字号未解析':>10}"
          f"{'C1粗':>6}{'C2粗短':>7}{'C3粗短无句读':>13}{'D1大字':>8}{'D2大短':>7}{'基准pt':>7}")
    print("-" * 118)
    for r in recs:
        if r.get("error"):
            print(f"{r['file'][:32]:<34}  [X] {r['error'][:60]}")
            continue
        base = r["baseline_size_pt"]
        print(f"{r['file'][:32]:<34}{r['n_paras']:>4}{r['n_heading']:>8}"
              f"{r['n_size_unresolved']:>10}"
              f"{r['c1_bold']:>6}{r['c2_bold_short']:>7}{r['c3_bold_short_nopunct']:>13}"
              f"{r['d1_big']:>8}{r['d2_big_short']:>7}"
              f"{(f'{base:g}' if base is not None else '—'):>7}")
    print("-" * 118)

    tot = {k: sum(r[k] for r in ok) for k in
           ("n_paras", "n_heading", "c1_bold", "c2_bold_short",
            "c3_bold_short_nopunct", "d1_big", "d2_big_short")}
    print(f"\n【汇总】{len(files)} 个 docx（成功 {len(ok)}）｜ 段 {tot['n_paras']}，"
          f"现判 heading **{tot['n_heading']}**")
    print("  格式候选（**分母都是「非 heading 段」，不是「全部段」**）：")
    print(f"    C1 全段加粗                      ：{tot['c1_bold']}")
    print(f"    C2 C1 且 ≤{SHORT_CHARS} 字符             ：{tot['c2_bold_short']}")
    print(f"    C3 C2 且不以句读结尾             ：{tot['c3_bold_short_nopunct']}")
    print(f"    D1 字号 > 本文件正文基准         ：{tot['d1_big']}")
    print(f"    D2 D1 且 ≤{SHORT_CHARS} 字符             ：{tot['d2_big_short']}")

    # ---- `2.0-46` 残留极短块 ↔ 格式判据 ----
    tinies = [(r["file"], t) for r in ok for t in r["tiny_chunks"]]
    print(f"\n【`2.0-46` 残留极短块】共 {len(tinies)} 个（阈值 <{TINY_CHARS} 字符，与 l0_report 同口径）")
    if tinies:
        print(f"  {'文件':<30}{'长度':>4} {'单段?':>6}{'现判heading':>11}{'全段加粗':>9}{'字号pt':>7}  文本")
        for f, t in tinies:
            flag = ("✅" if t["single_para"] else "（多段合并）")
            sz = f"{t['size_pt']:g}" if t["size_pt"] is not None else "—"
            print(f"  {f[:28]:<30}{t['len']:>4} {flag:>6}"
                  f"{str(t['is_heading']):>11}{str(t['bold_all']):>9}{sz:>7}"
                  f"  {t['text'][:40]}")
        n_single = sum(1 for _, t in tinies if t["single_para"])
        n_rescue = sum(1 for _, t in tinies if t["single_para"] and t["bold_all"])
        print(f"  → 能对上单个段落的 {n_single}/{len(tinies)}；"
              f"其中**全段加粗**（格式判据可捞回）的 **{n_rescue}/{len(tinies)}**")

    # ---- 候选明细（打印适可而止，全量进 JSON）----
    print("\n【C2 候选明细：非 heading + 全段加粗 + ≤%d 字符】" % SHORT_CHARS)
    shown = 0
    for r in ok:
        for p in r["cand_bold_short"]:
            if shown >= 60:
                break
            print(f"  {r['file'][:18]:<20} {p['size_pt'] if p['size_pt'] is not None else '—':>5}pt "
                  f"[{p['style'][:12]:<12}] {p['text'][:48]}")
            shown += 1
    if shown == 0:
        print("  （无）")
    elif sum(len(r["cand_bold_short"]) for r in ok) > shown:
        print(f"  …（另有 {sum(len(r['cand_bold_short']) for r in ok) - shown} 条，见 JSON）")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"params": {"chunk_size": s.chunk_size, "overlap": s.chunk_overlap,
                    "tiny_chars": TINY_CHARS, "short_chars": SHORT_CHARS},
         "summary": tot | {"files": len(files), "ok": len(ok),
                           "tiny_chunks": len(tinies),
                           "tiny_rescue_bold": sum(
                               1 for _, t in tinies if t["single_para"] and t["bold_all"])},
         "files_detail": recs},
        ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        shown_path: object = out.relative_to(ROOT)
    except ValueError:
        shown_path = out
    print(f"\n全量明细 → {shown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
