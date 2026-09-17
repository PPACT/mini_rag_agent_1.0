"""预判「精排的可救空间」：粗排把正确答案排在了第几位？

用途：决定"值不值得上更强的精排/本地 cross-encoder"。
逻辑：
  - 若正确答案多在粗排第 2~3 位 → 轻微排序调整即可救 → 换精排模型有戏
  - 若多在 11~20 位          → 需要大幅提升 → 精排模型很难做到
  - 若不在候选池(>20)        → 召回问题，任何精排都救不了（该改粗排/召回）

零成本：纯本地检索，不装模型、不调 LLM 生成。

用法：python eval/analyze_rerank_headroom.py [--n 80]
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
BUCKETS = [("第1位(已对)", lambda r: r == 1),
           ("第2-3位", lambda r: 2 <= r <= 3),
           ("第4-5位", lambda r: 4 <= r <= 5),
           ("第6-10位", lambda r: 6 <= r <= 10),
           ("第11-20位", lambda r: 11 <= r <= 20),
           ("不在候选池(>20)", lambda r: r is None)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]
    sem = asyncio.Semaphore(CONCURRENCY)
    ranks: list[int | None] = [None] * len(rows)

    async def one(i: int, r: dict) -> None:
        trace: dict = {}
        async with sem:
            await retrieve(r["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, trace=trace)
        for j, c in enumerate(trace.get("candidates", []), 1):
            if c.source_file == r["source"]:
                ranks[i] = j
                return

    await asyncio.gather(*[one(i, r) for i, r in enumerate(rows)])

    n = len(rows)
    print(f"{n} 道清晰题 —— 期望来源在【粗排候选池(20条)】中的位置\n")
    print(f"{'位置':<18}{'题数':<8}{'占比':<10}")
    print("-" * 38)
    for label, pred in BUCKETS:
        c = sum(1 for r in ranks if pred(r))
        print(f"{label:<18}{c:<8}{c / n:<10.1%}")

    # 粗排 top-1 错的题里，正确答案在哪
    wrong = [(rows[i]["question"], ranks[i]) for i in range(n) if ranks[i] != 1]
    print(f"\n粗排 top-1 未命中的 {len(wrong)} 道，其正确答案位置：")
    for q, r in wrong[:20]:
        print(f"  第{r if r else '>20':<4} {q[:30]}")

    # 结论
    easy = sum(1 for r in ranks if r is not None and 2 <= r <= 5)   # 微调可救
    hard = sum(1 for r in ranks if r is not None and r >= 6)        # 需大幅提升
    miss = sum(1 for r in ranks if r is None)                       # 召回问题
    print(f"\n【可救空间判断】")
    print(f"  第2-5位（轻微调整即可救）: {easy} 道  → 更强精排有机会")
    print(f"  第6-20位（需大幅提升）  : {hard} 道  → 精排模型很难")
    print(f"  不在候选池（召回问题）   : {miss} 道  → 任何精排都救不了，要改粗排")
    if easy + hard:
        print(f"\n  若精排能全部救回，上限提升: {(easy + hard) / n:.1%}（当前实际只救回 3 道）")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
