"""量化「LLM 精排」到底值不值：它改变了多少题的 top-1？改对了多少、改错了多少？

背景：用户观察"开/关精排答案没区别"。这很可能是**大多数题粗排就排对了**，
精排只在"粗排排错"时才有机会发挥作用。

方法：对每道清晰题跑一次带 trace 的检索，比较：
  - 粗排 top-1（RRF 融合后第一条）是否命中期望来源
  - 精排 top-1（精排后第一条）是否命中
输出变化矩阵 + 净收益。

用法：python eval/measure_rerank_value.py [--n 80]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from run_eval import EVAL_DEPARTMENT, EVAL_SECRET_LEVEL  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

DATASET = EVAL_DIR / "dataset_clear.jsonl"
CONCURRENCY = 5


def _hit_top1(chunks, source: str) -> bool:
    return bool(chunks) and chunks[0].source_file == source


def _rank(chunks, source: str) -> int | None:
    for i, c in enumerate(chunks, 1):
        if c.source_file == source:
            return i
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
    sem = asyncio.Semaphore(CONCURRENCY)

    stats = {"pre_hit": 0, "post_hit": 0, "changed": 0, "fixed": 0, "broke": 0, "same_wrong": 0}
    details: list[tuple] = []

    async def one(r: dict) -> None:
        trace: dict = {}
        async with sem:
            await retrieve(r["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, trace=trace)
        cand, final = trace.get("candidates", []), trace.get("final", [])
        pre = _hit_top1(cand, r["source"])
        post = _hit_top1(final, r["source"])
        if pre:
            stats["pre_hit"] += 1
        if post:
            stats["post_hit"] += 1
        if pre != post:
            stats["changed"] += 1
            if post and not pre:
                stats["fixed"] += 1
                details.append(("修复", r["question"], _rank(cand, r["source"]), 1))
            else:
                stats["broke"] += 1
                details.append(("误杀", r["question"], 1, _rank(final, r["source"])))
        elif not post:
            stats["same_wrong"] += 1

    await asyncio.gather(*[one(r) for r in rows])

    n = len(rows)
    print(f"{n} 道清晰题（top-1 是否命中期望来源）\n")
    print(f"  粗排 top-1 命中 : {stats['pre_hit']}/{n} = {stats['pre_hit'] / n:.1%}")
    print(f"  精排 top-1 命中 : {stats['post_hit']}/{n} = {stats['post_hit'] / n:.1%}")
    print(f"  ── 精排改变了 top-1 的题：{stats['changed']} 道 ({stats['changed'] / n:.1%})")
    print(f"      其中 改对（救回来）: {stats['fixed']}")
    print(f"      其中 改错（误杀）  : {stats['broke']}")
    print(f"      净收益             : {stats['fixed'] - stats['broke']:+d} 道")
    print(f"  两边都错（精排也没救）: {stats['same_wrong']}")

    if details:
        print("\n=== 明细 ===")
        for kind, q, a, b in details:
            print(f"  [{kind}] {q[:26]:<28} 粗排第{a} → 精排第{b}")

    print(f"\n解读：若「改变了 top-1」的比例很低（如 <10%），说明精排多数时候无事可做；")
    print(f"      此时它的价值 = 净收益 / 它消耗的时间（实测约 9 秒/次）。")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
