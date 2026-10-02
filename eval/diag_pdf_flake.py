"""`2.0-50` 第一步：**给 `03_*.pdf` 的间歇失败定性** —— 不修，先量。

## 为什么它是高优先（文档侧 `§1.55①`）

**它让 `ingest` 结果随运行而变** —— 同一份代码、同一批语料，**跑两次得到不同的库**。
→ **任何 R@K / 任何对比都是在流沙上做的**；
→ 更直接的是：**`2.0-34`（增量重跑）做完后没法验证** ——
   跑两次结果不一样，**分不清是"重跑错了"还是"flake 又犯了"**。

## 它回答什么

| 问题 | 怎么看 |
|---|---|
| **失败率多少** | N 次里失败几次 |
| **失败是随机的还是确定的** | 失败在序列里的分布（聚在开头？隔一次？） |
| **是 pdfplumber 的已知问题，还是我们的用法问题** | **栈的**形状 + **异常类型/消息** |
| **是不是并发/状态相关** | 同一进程内连续跑 vs 每次**新进程**，两组对比 |

⚠️ **只解析、不嵌入** —— 栈明确在 `pdfplumber.extract_text`，
所以**不必付 30 次嵌入的代价**去复现它（真要复现整条 ingest 再说）。

用法：
    python eval/diag_pdf_flake.py                      # 默认 30 次
    python eval/diag_pdf_flake.py --runs 50 --file docs/corpus/03_考勤与休假管理规定.pdf
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.document_parser.routing import parse_document  # noqa: E402

DEFAULT_FILE = ROOT / "docs" / "corpus" / "03_考勤与休假管理规定.pdf"


def _signature(exc: BaseException) -> str:
    """失败的**指纹**：只留"哪一类错、在哪几行"。

    ⚠️ **不把 traceback 全文当指纹** —— 里面有内存地址/行号变体，会把同一种失败算成很多种。
    """
    tb = traceback.extract_tb(exc.__traceback__)
    tail = [f"{Path(f.filename).name}:{f.lineno}" for f in tb[-4:]]
    return f"{type(exc).__name__}: {exc} @ " + " <- ".join(tail)


def _run_once(path: Path) -> tuple[bool, str | None, str | None]:
    """跑一次真实解析入口。返回 (成功?, 指纹, 完整 traceback)。"""
    try:
        doc = parse_document(str(path))
        return (bool(doc.blocks), None, None)
    except Exception as e:  # noqa: BLE001 —— 诊断脚本就是要**接住一切**
        return (False, _signature(e), traceback.format_exc())


def _run_in_fresh_process(path: Path) -> tuple[bool, str | None]:
    """**新进程**跑一次 —— 用来分辨"是不是进程内状态/缓存造成的"。"""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from src.document_parser.routing import parse_document\n"
        "try:\n"
        "    parse_document(r'%s'); print('OK')\n"
        "except Exception as e:\n"
        "    print('FAIL', type(e).__name__, e)\n"
    ) % (ROOT, path)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    out = (r.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    return (last.startswith("OK"), None if last.startswith("OK") else last[:160])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    ap.add_argument("--json", default=str(ROOT / "logs" / "diag_pdf_flake.json"))
    ap.add_argument("--fresh-procs", type=int, default=10,
                    help="额外用**新进程**跑几次（分辨进程内状态影响）")
    ap.add_argument("--corpus-runs", type=int, default=0,
                    help="额外：把 docs/corpus **整批**解析 N 遍 —— 贴近真实的 ingest 条件")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"[X] 文件不存在：{path}")
        return 1

    print(f"2.0-50 诊断 ｜ {path.name} ｜ 同进程连续 {args.runs} 次 + 新进程 {args.fresh_procs} 次")
    print("⚠️ 只解析、不嵌入（栈在 pdfplumber.extract_text，不必付嵌入代价）\n")

    seq: list[bool] = []
    sigs: Counter[str] = Counter()
    first_tb: str | None = None
    for i in range(1, args.runs + 1):
        ok, sig, tb = _run_once(path)
        seq.append(ok)
        if not ok:
            sigs[sig] += 1
            first_tb = first_tb or tb
        print(f"  #{i:>3} {'✅' if ok else '❌'}", end="\n" if (i % 10 == 0 or not ok) else " ")
    print()

    n_fail = seq.count(False)
    print(f"【同进程】失败 {n_fail}/{args.runs} = {n_fail / args.runs * 100:.1f}%")
    if n_fail:
        # 失败是不是聚在开头？那说明有"预热"效应
        idx = [i for i, ok in enumerate(seq) if not ok]
        print(f"  失败发生在第 {idx} 次（0-based）｜ 前 5 次里失败 {sum(1 for i in idx if i < 5)} 次")
        print(f"  失败指纹（{len(sigs)} 种）：")
        for s, n in sigs.most_common(5):
            print(f"    ×{n}  {s}")

    fresh = [_run_in_fresh_process(path) for _ in range(args.fresh_procs)]
    nf = sum(1 for ok, _ in fresh if not ok)
    print(f"\n【新进程】失败 {nf}/{args.fresh_procs} = {nf / args.fresh_procs * 100:.1f}%")
    if nf:
        for _, msg in fresh:
            if msg:
                print(f"    {msg}")

    # ---- 整批模式：贴近真实的 ingest（顺序解析一整批，进程内状态会累积）----
    corpus_fail: Counter[str] = Counter()
    corpus_total: Counter[str] = Counter()
    corpus_sigs: Counter[str] = Counter()
    if args.corpus_runs:
        files = sorted(p for p in (ROOT / "docs" / "corpus").iterdir() if p.is_file())
        print(f"\n【整批模式】docs/corpus 全量解析 {args.corpus_runs} 遍（{len(files)} 个文件/遍）")
        for r in range(1, args.corpus_runs + 1):
            for f in files:
                corpus_total[f.name] += 1
                ok, sig, tb = _run_once(f)
                if not ok:
                    corpus_fail[f.name] += 1
                    if sig:
                        corpus_sigs[sig] += 1
                    if not first_tb:
                        first_tb = tb
            print(f"  第 {r} 遍完成（累计失败 {sum(corpus_fail.values())}）")
        if corpus_fail:
            print("  逐文件失败：")
            for name, n in corpus_fail.most_common():
                print(f"    {name}  {n}/{corpus_total[name]}")
            print(f"  失败指纹（{len(corpus_sigs)} 种）：")
            for s, n in corpus_sigs.most_common(5):
                print(f"    ×{n}  {s}")
        else:
            print(f"  ⚠️ 整批 {args.corpus_runs} 遍**一次都没失败**")

    if first_tb:
        print("\n【首次失败的完整栈（尾部 18 行）】")
        for line in first_tb.strip().splitlines()[-18:]:
            print("  " + line)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "file": path.name, "runs": args.runs, "failures": n_fail,
        "fail_rate": n_fail / args.runs,
        "sequence": ["ok" if ok else "fail" for ok in seq],
        "signatures": dict(sigs),
        "fresh_procs": {"runs": args.fresh_procs, "failures": nf},
        "corpus": {"runs": args.corpus_runs,
                   "failures": dict(corpus_fail),
                   "totals": dict(corpus_total)},
        "first_traceback": first_tb,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        print(f"\n产物 → {out.relative_to(ROOT)}")
    except ValueError:
        print(f"\n产物 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
