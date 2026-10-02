"""`2.0-29` 复核：PDF 抽文本时，**数字和它的标签是不是被拆散了**。

## 背景（`改进计划.md` §2.0-29）

参考文档点名「**不要直接用 `pdfplumber` 按物理顺序抽文本**」，并推荐 MinerU / Docling 等**重依赖**。
⚠️ 计划里写明：**不要照单全收，先用真实语料证明必要性**（`P-4` / `P-5`）。

## 它量什么

1. 这些 PDF **到底有没有表格结构**（`extract_tables()` / 线 / 矩形）
2. 抽出来的文本里，**数字是不是被切成了独立行**（那样它就和标签分离了）
3. 改 `y_tolerance` 能不能修好 —— **以及会不会把别的文档弄坏**

⚠️ **判据是"成对出现的指标"**：只看"孤立数字行变少"会被**过度合并**骗到 ——
所以同时看 **`第 条`（该有的是 `第 N 条`）** 和 **一字一行**（合并崩了的信号）。
⭐ 实测就抓到过：`y_tolerance=12` 修好了 4 份，却把第 5 份**拆成了一字一行**。

用法：
    python eval/probe_pdf_text_order.py
    python eval/probe_pdf_text_order.py --dir docs/local/corpus --json logs/pdf_text_order.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.paths import CORPUS_DIR  # noqa: E402

import pdfplumber  # noqa: E402

TOLS = (3, 8, 12, 16)


def _stats(text: str) -> dict:
    """一份文本的四个数 —— **成对看才有意义**。"""
    lines = text.splitlines()
    return {
        "lines": len(lines),
        "lone_number_lines": sum(1 for ln in lines if re.fullmatch(r"\s*[\d.,%]+\s*", ln)),
        # 「第 条」= 数字被搬走了；「第 N 条」= 完整 —— 两个一起看
        "broken_article": len(re.findall(r"第\s+条", text)),
        "ok_article": len(re.findall(r"第\s*\d+\s*条", text)),
        # ⚠️ 过度合并的信号：整行只剩一个字
        "one_char_lines": sum(1 for ln in lines if len(ln.strip()) == 1),
    }


def _scan(path: Path) -> dict:
    rec: dict = {"file": path.name, "by_tolerance": {}}
    with pdfplumber.open(str(path)) as pdf:
        rec["pages"] = len(pdf.pages)
        try:
            rec["extract_tables"] = sum(len(p.extract_tables() or []) for p in pdf.pages)
        except Exception as e:  # noqa: BLE001
            rec["extract_tables"] = f"{type(e).__name__}: {e}"
        # 表格的**物理痕迹**（线 / 矩形）—— 没有它，"表格"就只是排版上的对齐
        rec["lines_and_rects"] = sum(len(p.lines) + len(p.rects) for p in pdf.pages)
        for tol in TOLS:
            text = "\n".join((p.extract_text(y_tolerance=tol) or "") for p in pdf.pages)
            rec["by_tolerance"][str(tol)] = _stats(text)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(CORPUS_DIR))
    ap.add_argument("--json", default=str(ROOT / "logs" / "pdf_text_order.json"))
    args = ap.parse_args()

    files = sorted(p for p in Path(args.dir).glob("*.pdf"))
    if not files:
        print(f"[X] {args.dir} 下没有 .pdf")
        return 1

    print(f"2.0-29 ｜ PDF 文本顺序 ｜ {len(files)} 份 ｜ y_tolerance ∈ {TOLS}\n")
    recs = [_scan(p) for p in files]

    print(f"{'文件':<26}{'页':>3}{'表':>4}{'线/框':>6}   "
          + "".join(f"{'y=' + str(t):>26}" for t in TOLS))
    print("-" * 118)
    for r in recs:
        head = f"{r['file'][:24]:<26}{r['pages']:>3}{str(r['extract_tables']):>4}{r['lines_and_rects']:>6}   "
        cells = ""
        for t in TOLS:
            s = r["by_tolerance"][str(t)]
            cells += f"{s['lone_number_lines']:>4}/{s['broken_article']:>3}/{s['ok_article']:>3}{'⚠️' if s['one_char_lines'] > 20 else '  '}"
        print(head + cells)
    print("-" * 118)
    print("  每格 = 孤立数字行 / 「第 条」(该是「第 N 条」) / 「第N条」 ｜ ⚠️ = 一字一行（合并崩了）")

    tot = {t: {k: sum(r["by_tolerance"][str(t)][k] for r in recs)
               for k in ("lone_number_lines", "broken_article", "ok_article", "one_char_lines")}
           for t in TOLS}
    print("\n【合计】")
    for t in TOLS:
        s = tot[t]
        print(f"  y_tol={t:>2}: 孤立数字行 {s['lone_number_lines']:>3} ｜ 「第 条」{s['broken_article']:>3} "
              f"｜ 「第N条」{s['ok_article']:>3} ｜ 一字一行 {s['one_char_lines']:>3}"
              + ("  ← 现在用的" if t == 3 else ""))

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tolerances": list(TOLS), "files": recs, "totals": tot},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        shown: object = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\n产物 → {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
