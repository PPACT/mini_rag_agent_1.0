"""测量「缩短歧义判定 prompt」的收益与代价。

背景：歧义判定是最大延迟瓶颈（1.8~2.3s / 56.7%），门控方案已被数据排除。
剩下可试的：**缩短送给判定 LLM 的候选文本**（`ambiguity_snippet_chars`，当前 300 字）。
更短的 prompt → LLM 处理更少 token → 更快，但**可能影响判定质量**。

方法（控制变量）：**候选池固定**（只检索一次），仅改变 snippet 长度，
对每档记录：① 平均耗时 ② 漏报率 ③ 误报率 ④ 与 300 字基准的判定一致率。

用法：python eval/measure_ambiguity_snippet.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.rag.ambiguity import check_ambiguity  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

LEVELS = (300, 150, 100, 60)     # snippet 字数档位（300 是当前值，作为基准）
N_CLEAR = 20                     # 清晰题取样（应判"不歧义"）
N_AMBIG = 15                     # 歧义题取样（应判"歧义"）
N_ADV = 10                       # 对抗题取样（应判"歧义"）
DEPTS = ["IT", "公司"]
LEVEL = 3


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


async def main() -> None:
    s = get_settings()
    clear = _load("dataset_clear.jsonl")[:N_CLEAR]
    clar = _load("dataset_clarify.jsonl")
    ambiguous = [r for r in clar if r.get("kind") == "ambiguous"][:N_AMBIG]
    adversarial = [r for r in clar if r.get("kind") == "adversarial"][:N_ADV]

    # (问题, 期望是否歧义)
    cases = ([(r, False) for r in clear]
             + [(r, True) for r in ambiguous]
             + [(r, True) for r in adversarial])
    print(f"题集：不歧义 {len(clear)} / 歧义 {len(ambiguous)} / 对抗 {len(adversarial)}"
          f" = {len(cases)} 题，{len(LEVELS)} 档 snippet\n", flush=True)

    # ① 预先检索候选池（固定，避免重跑检索引入噪声）
    print("检索候选池（只做一次）...", flush=True)
    pool: dict[str, list] = {}
    for r, _ in cases:
        _, chunks = await retrieve(r["question"], DEPTS, LEVEL, top_k=5)
        pool[r["id"]] = chunks

    # ② 逐档测试
    base_verdicts: dict[str, bool] = {}
    rows = []
    for snippet in LEVELS:
        s.ambiguity_snippet_chars = snippet
        lat, verdicts = [], {}
        for r, should_ambig in cases:
            t0 = time.perf_counter()
            res = await check_ambiguity(r["question"], pool[r["id"]])
            lat.append((time.perf_counter() - t0) * 1000)
            verdicts[r["id"]] = res.ambiguous
        if snippet == LEVELS[0]:
            base_verdicts = verdicts

        # 统计
        miss = sum(1 for r, sh in cases if sh and not verdicts[r["id"]])
        false_pos = sum(1 for r, sh in cases if not sh and verdicts[r["id"]])
        n_amb = sum(1 for _, sh in cases if sh)
        n_clr = len(cases) - n_amb
        agree = sum(1 for k in verdicts if verdicts[k] == base_verdicts[k]) / len(cases)
        rows.append({
            "snippet": snippet,
            "avg_ms": sum(lat) / len(lat),
            "miss": miss / n_amb if n_amb else 0,
            "fp": false_pos / n_clr if n_clr else 0,
            "agree": agree,
        })
        print(f"  snippet={snippet:>3}: 平均 {rows[-1]['avg_ms']:>6.0f}ms  "
              f"漏报 {rows[-1]['miss']:>5.1%}  误报 {rows[-1]['fp']:>5.1%}  "
              f"与基准一致 {agree:>5.1%}", flush=True)

    print(f"\n{'snippet':<10}{'平均耗时':<12}{'漏报率':<10}{'误报率':<10}{'与300一致':<10}")
    print("-" * 52)
    for r in rows:
        print(f"{r['snippet']:<10}{r['avg_ms']:<12.0f}{r['miss']:<10.1%}{r['fp']:<10.1%}{r['agree']:<10.1%}")

    print("\n解读：找「耗时明显下降 且 漏报/误报不明显变差」的档位。")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
