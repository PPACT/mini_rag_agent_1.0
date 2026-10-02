"""`2.0-48` **shadow** —— 新判据「整段加粗」相对旧判据改到了什么？（**只读 / 零成本 / 不重嵌**）

## 为什么先 shadow（文档侧 `交流区 §1.40②`）

不让直接上的三条理由：① 4 篇 docx 会从"0 章节"变"有章节" → 要重嵌；
② "肉眼看零误报"是 6 篇样本的观察、**不是闸**；③ 现在**拿不到"重嵌范围"这个数**。

所以要量三个数：

| # | 量什么 | 为什么 |
|---|---|---|
| ① | **章节数会变成多少** | 会不会凭空造出章节 |
| ② | ⭐ **`content` 会变多少个块** | **这才是重嵌成本** |
| ③ | **认了多少 / 其中多少真像标题** | 判据的区分度 |

## 它的 A/B 是哪两条路（`2.0-48` 开启后已翻面）

**同一个判据实现，一处定义**（`parsers.heading_rule_style_or_bold`）：

- **基线**：`parse_docx(path, heading_rule=_is_heading)` —— **旧判据**（只认样式名）
  ⚠️ 必须**显式**传：`parse_docx` 的默认**已经切成新判据**了，不传就两边一样、差值恒为 0
- **shadow**：`parse_docx(path, heading_rule=heading_rule_style_or_bold)` —— **生产现口径**
- ⭐ **自检**：生产入口 `routing.parse_document(path)` 必须与 shadow 那条链**逐块相同**
  （P-2：核实"打算用的"＝"实际生效的"）

→ 三个数因此是 **"旧判据 → 新判据"的差**，与开启前跑出来的那一轮**可直接对照**。

## ⚠️ 两条口径说明（别把数读错）

1. **`content` 变化只可能来自"紧邻标题的表格补 caption"** ——
   `_ATOMIC` 只有 `table`/`code`，而文本段的拼接（`"\n".join(...)`）与 `kind` **无关**；
   `split_text` 本来就用它自己的「短行 + 无句读」判据认标题、只设 `title` 不改文本。
   → 所以本探针的"新增块"基本就是**被补了 caption 的表格块**。
2. **"重嵌范围" = 新增块数**（要重新嵌入的文本），**不是**"变化块数"——
   消失的块只需删行、不必嵌入。两个数都报，但**成本项是前者**。

用法：
    python eval/shadow_bold_heading.py
    python eval/shadow_bold_heading.py --dir docs/local/corpus --json logs/shadow_bold_heading.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.paths import CORPUS_DIR  # noqa: E402

from src.config.settings import get_settings  # noqa: E402
from src.document_parser.chunking import split_blocks  # noqa: E402
from src.document_parser.parsers import (  # noqa: E402
    _is_heading,
    heading_rule_style_or_bold,
    parse_docx,
)
from src.document_parser.routing import parse_document  # noqa: E402
from src.document_parser.table_recovery import recover_tables  # noqa: E402

OUT_DEFAULT = ROOT / "logs" / "shadow_bold_heading.json"
SHORT_CHARS = 40          # 与 parser / splitter 同款的"像标题"展示宽度（非判据）
_END_PUNCT = "。！？；;.!?，,：:"


def _same_blocks(a, b) -> bool:
    """两套 blocks 是否逐块（kind, text）相同 —— 用来钉住 AB 台子没跑偏。"""
    ka = [(x.kind, x.text) for x in a.blocks]
    kb = [(x.kind, x.text) for x in b.blocks]
    return ka == kb


def _looks_like_title(text: str) -> bool:
    """⚠️ **只是展示用的代理指标**，不是判据本体。

    取值恰好复刻 `semantic_splitter._is_heading` 的形态条件（短行 + 不以句读结尾），
    这样"像标题"这一列**能与切块层已有的行为对照**。**别把它当成第二道闸。**
    """
    t = text.strip()
    return bool(t) and len(t) <= SHORT_CHARS and t[-1] not in _END_PUNCT


def _measure(path: Path, chunk_size: int, overlap: int) -> dict:
    rec: dict = {"file": path.name}
    try:
        # `2.0-48` 开启后，`parse_docx` 的**默认已是新判据** →
        # 基线必须**显式**传旧判据，否则两边同为新判据、差值恒为 0（台子就白跑了）。
        base_doc, _ = recover_tables(parse_docx(str(path), heading_rule=_is_heading))
        # shadow = 生产现口径。⚠️ 必须**走同一条后处理** ——
        # `parse_document` = 路由 → 解析器 → `recover_tables`；少了 `recover_tables`，
        # 伪表格会被当成普通段落 → AB 两组**差了两个变量**（P-9）。
        shadow_doc, _ = recover_tables(
            parse_docx(str(path), heading_rule=heading_rule_style_or_bold))
        # ⭐ 自检：生产入口必须与 shadow 那条链**逐块相同** ——
        # 不等 = 台子测的不是实际跑的径，数字一律不可信（实测踩过：漏了 recover_tables）。
        rec["harness_ok"] = _same_blocks(shadow_doc, parse_document(str(path)))
        base_chunks = split_blocks(base_doc, chunk_size, overlap)
        shadow_chunks = split_blocks(shadow_doc, chunk_size, overlap)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    base_heads = [b.text for b in base_doc.blocks if b.kind == "heading"]
    shadow_heads = [b.text for b in shadow_doc.blocks if b.kind == "heading"]
    new_heads = list((Counter(shadow_heads) - Counter(base_heads)).elements())

    cb = Counter(c.text for c in base_chunks)
    ca = Counter(c.text for c in shadow_chunks)
    added = ca - cb        # 要**新增/重嵌**的块
    removed = cb - ca      # 要**删行**的块

    rec.update(
        n_heads_before=len(base_heads),
        n_heads_after=len(shadow_heads),
        n_new_heads=len(new_heads),
        new_heads=new_heads,
        n_new_like_title=sum(1 for t in new_heads if _looks_like_title(t)),
        n_chunks_before=len(base_chunks),
        n_chunks_after=len(shadow_chunks),
        n_reembed=sum(added.values()),
        n_removed=sum(removed.values()),
        titles_before=sorted({c.title for c in base_chunks if c.title}),
        titles_after=sorted({c.title for c in shadow_chunks if c.title}),
        added_samples=[{"text": t[:160], "count": n} for t, n in added.most_common(6)],
        removed_samples=[{"text": t[:80], "count": n} for t, n in removed.most_common(6)],
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(CORPUS_DIR))
    ap.add_argument("--json", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    s = get_settings()
    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"[X] 目录不存在：{src_dir}")
        return 1
    files = sorted(p for p in src_dir.iterdir()
                   if p.is_file() and p.suffix.lower() == ".docx")
    if not files:
        print(f"[X] {src_dir} 下没有 .docx")
        return 1

    recs = [_measure(p, s.chunk_size, s.chunk_overlap) for p in files]
    ok = [r for r in recs if not r.get("error")]

    print(f"2.0-48 shadow ｜ A/B = 旧判据(只认样式) → 生产现判据(样式 OR 整段加粗) ｜ {src_dir}")
    print(f"参数：chunk_size={s.chunk_size}  overlap={s.chunk_overlap} ｜ "
          f"⚠️ 只读不写：本脚本不嵌入、不改库\n")

    print(f"{'文件':<34}{'heading 前→后':>14}{'新认':>6}{'像标题':>7}"
          f"{'块 前→后':>12}{'新增块':>7}{'删除块':>7}{'章节标题 前→后':>15}")
    print("-" * 112)
    for r in recs:
        if r.get("error"):
            print(f"{r['file'][:32]:<34}  [X] {r['error'][:60]}")
            continue
        heads = f"{r['n_heads_before']}→{r['n_heads_after']}"
        chunks = f"{r['n_chunks_before']}→{r['n_chunks_after']}"
        titles = f"{len(r['titles_before'])}→{len(r['titles_after'])}"
        print(f"{r['file'][:32]:<34}{heads:>14}{r['n_new_heads']:>6}"
              f"{r['n_new_like_title']:>7}{chunks:>12}"
              f"{r['n_reembed']:>7}{r['n_removed']:>7}{titles:>15}")
    print("-" * 112)

    tot = {k: sum(r[k] for r in ok) for k in
           ("n_heads_before", "n_heads_after", "n_new_heads", "n_new_like_title",
            "n_chunks_before", "n_chunks_after", "n_reembed", "n_removed")}
    print(f"\n【汇总 · {len(files)} 篇 docx，成功 {len(ok)}】")
    print(f"  ① 章节数（heading 块）：**{tot['n_heads_before']} → {tot['n_heads_after']}**")
    print(f"  ② ⭐ content 变化（= 重嵌成本）：**新增/重嵌 {tot['n_reembed']} 块**"
          f" ｜ 删除 {tot['n_removed']} 块"
          f" ｜ 块总数 {tot['n_chunks_before']} → {tot['n_chunks_after']}")
    print(f"  ③ 认了多少：**新增标题 {tot['n_new_heads']} 条**"
          f" ｜ 其中「像标题」（短且无句读，**代理指标**）：{tot['n_new_like_title']}")

    # ⭐ AB 台子自检：复算的"现判据 + 同一后处理"必须与生产基线逐块相同
    bad_harness = [r["file"] for r in ok if not r.get("harness_ok")]
    if bad_harness:
        print(f"\n  [X] ⚠️ **AB 台子失真** —— {len(bad_harness)} 篇的复算 ≠ 生产基线："
              f"{', '.join(bad_harness[:4])}")
        print("      → 上面所有数字**不可信**（多半是 shadow 少了某道后处理）")
    else:
        print(f"\n  ✅ AB 台子自检通过：{len(ok)}/{len(ok)} 篇的**生产现判据那条链**"
              f"与 `parse_document` 逐块一致（台子测的确实是实际跑的径）")

    print("\n【新增的标题（全量）】")
    for r in ok:
        for t in r["new_heads"]:
            mark = "  " if _looks_like_title(t) else "⚠️"
            print(f"  {mark} {t[:60]}")
    if tot["n_new_heads"] == 0:
        print("  （无）")

    print("\n【content 新增块 样例（= 重嵌对象）】")
    shown = 0
    for r in ok:
        for a in r["added_samples"]:
            if shown >= 8:
                break
            print(f"  {r['file'][:18]:<20} ×{a['count']}  {a['text'][:96]!r}")
            shown += 1
    if shown == 0:
        print("  （无 —— 说明没有表格被补上 caption）")

    # ---- ⚠️ 验收③ 的反向用例扫描：语料里有没有"整段加粗但不是标题"的强调句 ----
    bad = [r["file"] + " ｜ " + t for r in ok for t in r["new_heads"]
           if not _looks_like_title(t)]
    print(f"\n【反向用例扫描】整段加粗但**不像标题**的段落：**{len(bad)}** 条")
    for b in bad[:10]:
        print(f"  ⚠️ {b}")
    if not bad:
        print("  → 本语料**找不到反例**。⚠️ 但**不等于判据安全** —— "
              "反例只证明「这批没踩到」，机制由 tests/test_shadow_bold_heading.py 的合成长句兜底。")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"params": {"chunk_size": s.chunk_size, "overlap": s.chunk_overlap,
                    "short_chars": SHORT_CHARS},
         "summary": tot | {"files": len(files), "ok": len(ok),
                           "counterexamples": len(bad)},
         "files_detail": recs},
        ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        shown_path: object = out.relative_to(ROOT)
    except ValueError:
        shown_path = out
    print(f"\n逐文件明细 → {shown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
