"""解读 snippet 测量的逐题明细：区分「摇摆」与「稳定判错」。

背景：`measure_ambiguity_snippet.py` 的汇总表**不单调**，不能直接下结论：
  - p50 耗时：300→3219ms，150→3394ms（**反而更慢**），100→2725ms，60→2376ms
  - 漏报率：66.7% → 62.5% → 54.2% → **79.2%**（60 字最差，但 100 字最好）
非单调 = 差异很可能被噪声主导，而不是 snippet 长度主导。

本脚本按**题目**维度拆开，把误差分成两类：
  - **稳定判错**：4 档全部与期望不符 → snippet 调参救不了（能力/标注问题）
  - **摇摆**    ：档位之间翻转        → 噪声来源
若"摇摆"占比高，说明真正的问题不是 snippet 长度，而是判定本身不稳定。

⚠️ 本实验**缺少「同档位重测」这条噪声基线**（300 档的"一致率 100%"是自己跟自己比，
不含信息）。所以"摇摆"不能全部归因于 snippet——**下次要补跑一次 300 档重测**才能定量。

用法：python eval/analyze_snippet_result.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT))

DETAIL = ROOT / "logs" / "snippet_measure.json"
N_CLEAR, N_AMBIG, N_ADV = 20, 15, 10


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    if not DETAIL.exists():
        print(f"❌ 找不到 {DETAIL}，先跑 measure_ambiguity_snippet.py")
        return
    detail = json.loads(DETAIL.read_text(encoding="utf-8"))["detail"]
    levels = sorted(detail, key=lambda x: -int(x))     # 300 → 60（收缩顺序，便于读数）

    clar = _load("dataset_clarify.jsonl")
    cases = (
        [(r, False, "不歧义") for r in _load("dataset_clear.jsonl")[:N_CLEAR]]
        + [(r, True, "歧义") for r in clar if r.get("kind") == "ambiguous"][:N_AMBIG]
        + [(r, True, "对抗") for r in clar if r.get("kind") == "adversarial"][:N_ADV]
    )

    print(f"档位：{' / '.join(levels)}（字）\n")
    print(f"{'组':<8}{'题数':<6}{'稳定判对':<10}{'稳定判错':<10}{'摇摆':<8}{'摇摆题（档位判定序列）'}")
    print("-" * 78)

    wobble_ids: list[str] = []
    hard_wrong: list[str] = []
    for group in ("不歧义", "歧义", "对抗"):
        ok = wrong = wob = 0
        seqs = []
        for r, expect, g in cases:
            key = str(r["id"])                     # dataset 里 id 是 int，JSON 键是 str
            if g != group or key not in detail[levels[0]]:
                continue
            verdicts = [detail[lv][key]["ambiguous"] for lv in levels]
            if len(set(verdicts)) > 1:
                wob += 1
                seqs.append(f"{r['id']}=" + "".join("T" if v else "F" for v in verdicts))
                wobble_ids.append(r["id"])
            elif verdicts[0] == expect:
                ok += 1
            else:
                wrong += 1
                hard_wrong.append(f"{r['id']}({group})")
        total = ok + wrong + wob
        print(f"{group:<8}{total:<6}{ok:<10}{wrong:<10}{wob:<8}{' '.join(seqs[:6])}")

    kept = sum(1 for r, _, _ in cases if str(r["id"]) in detail[levels[0]])
    print(f"\n合计 {kept} 题：摇摆 {len(wobble_ids)}（{len(wobble_ids)/kept:.1%}）"
          f" ｜ 稳定判错 {len(hard_wrong)}（{len(hard_wrong)/kept:.1%}）")

    # 同一题在各档的耗时差（看是否题目难度主导而非 snippet）
    print("\n各档耗时分布（中位数，ms）：")
    for lv in levels:
        ms = sorted(v["ms"] for v in detail[lv].values())
        print(f"  {lv:>3}字: p10 {ms[len(ms)//10]:>6.0f}  p50 {ms[len(ms)//2]:>6.0f}  "
              f"p90 {ms[int(len(ms)*0.9)]:>6.0f}  max {ms[-1]:>6.0f}")

    if hard_wrong:
        print(f"\n稳定判错题（调 snippet 无用，需查能力/标注）：{' '.join(hard_wrong[:20])}")

    # 长尾诊断：平均(4.6s)远高于中位数(3.2s) → 尾部在拖平均。
    # 判据：同一题「各档都慢」= 系统性（如该题候选文本特别长）；
    #       各档慢的不是同一批 = 随机 API 抖动，与 snippet 无关。
    print("\n长尾诊断（单次 >6s 的调用）：")
    slow_sets = {}
    for lv in levels:
        slow = {k for k, v in detail[lv].items() if v["ms"] > 6000}
        slow_sets[lv] = slow
        ids = " ".join(sorted(slow)[:8])
        print(f"  {lv:>3}字: {len(slow):>2} 次" + (f"   → {ids}" if slow else ""))
    union = set().union(*slow_sets.values()) if slow_sets else set()
    inter = set.intersection(*slow_sets.values()) if slow_sets else set()
    print(f"  并集 {len(union)} 题（至少慢一次）｜ 交集 {len(inter)} 题（各档都慢）")
    if len(union) and len(inter) / len(union) < 0.5:
        print("  → 交集远小于并集：**随机抖动**，与 snippet 长度无关")
    elif inter:
        print("  → 交集接近并集：**系统性的**（同一题反复慢，查该题候选长度）")

    print("\n结论判据：")
    print("  · 摇摆占比高（>15%）→ 差异主要是噪声，snippet 不是有效杠杆")
    print("  · 稳定判错占比高     → 是能力问题，缩短 snippet 只会更糟")
    print("  · 两者都低 + p50 单调下降 → 才值得调小 snippet")


if __name__ == "__main__":
    main()
