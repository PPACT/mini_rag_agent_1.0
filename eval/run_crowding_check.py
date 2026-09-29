"""诊断：MMR 是否值得做？——失败主因是「近似块挤占」还是「检索盲区」？

MMR 只解决"期望来源在候选池里，但被同主题近似块挤出了 top-k"。
若失败主因是"期望来源根本不在候选池"（top-20 都进不了），MMR 无用，应跳过。

方法（纯向量 + 混合，关精排看候选池原貌）：
  对鲁棒集每道清晰题，取 top-20 候选，看期望来源的位置：
  - 在 top-5   → 检索可达（命中）
  - 在 6~20   → 被近似块挤占（MMR 潜在有用）★
  - 不在 top-20 → 检索盲区（MMR 无用）★

用法：python eval/run_crowding_check.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from run_eval import EVAL_DEPARTMENT, EVAL_SECRET_LEVEL  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

DATASET = EVAL_DIR / "dataset_clear.jsonl"
POOL_K = 20   # 候选池大小（与生产 rerank_candidates 一致）
TOPK = 5      # 最终返回数
CONCURRENCY = 5


async def main() -> None:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(rows)} 道清晰题，候选池 top-{POOL_K}，期望来源位置\n", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    stat = {"top5": 0, "crowd": 0, "blind": 0}
    crowd_detail: list[str] = []
    blind_detail: list[str] = []

    async def one(row: dict) -> None:
        async with sem:
            # 关精排：看候选池原貌（RRF 融合后的前 POOL_K）
            _, chunks = await retrieve(
                row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, kb=KB_STRESS,
                use_rerank=False, top_k=POOL_K,
            )
        positions = [i for i, c in enumerate(chunks, 1) if c.source_file == row["source"]]
        if positions and positions[0] <= TOPK:
            stat["top5"] += 1
        elif positions:
            stat["crowd"] += 1
            crowd_detail.append(f"#{row['id']} {row['question'][:22]} 期望来源排第 {positions[0]}")
        else:
            stat["blind"] += 1
            blind_detail.append(f"#{row['id']} {row['question'][:22]}")

    await asyncio.gather(*[one(r) for r in rows])

    n = len(rows)
    print(f"{'类别':<16}{'题数':<8}{'占比':<10}")
    print("-" * 34)
    print(f"{'命中(top-' + str(TOPK) + ')':<16}{stat['top5']:<8}{stat['top5'] / n:<10.1%}")
    print(f"{'挤占(6~' + str(POOL_K) + ')':<16}{stat['crowd']:<8}{stat['crowd'] / n:<10.1%}  ← MMR 潜在战场")
    print(f"{'盲区(>top-' + str(POOL_K) + ')':<16}{stat['blind']:<8}{stat['blind'] / n:<10.1%}  ← MMR 无用")

    print(f"\n=== 挤占明细（{len(crowd_detail)} 条，MMR 可能救）===")
    for d in crowd_detail[:15]:
        print(f"  {d}")
    print(f"\n=== 盲区明细（{len(blind_detail)} 条，MMR 救不了）===")
    for d in blind_detail[:15]:
        print(f"  {d}")

    print(f"\n结论：若『挤占』占比显著 → MMR 值得做；若『盲区』为主 → MMR 收益有限，应做召回侧。")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
