"""改述鲁棒性评测：同一问题换问法，结果是否稳定。

直接回应"换个问法就崩"——指标是**一致性**而非单次召回：
- **全命中一致率**：3 个问法（原问 + 2 改述）全部命中期望答案的比例
- **Top1 来源一致率**：3 个问法召回的 top-1 是否来自同一文档

用法（需已生成 dataset_paraphrase.jsonl + 语料已灌入）：
    python eval/gen_paraphrase.py     # 先生成改述
    python eval/run_paraphrase_eval.py --top 5
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

from run_eval import EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, first_hit_rank  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

DATASET = EVAL_DIR / "dataset_paraphrase.jsonl"
CONCURRENCY = 5


def load_rows() -> list[dict]:
    return [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _retrieve_one(q: str, top: int, sem: asyncio.Semaphore) -> tuple[list[str], str | None]:
    """返回 (top-k 来源列表, top-1 来源)。走生产配置（混合+精排，改写已关）。"""
    async with sem:
        _, chunks = await retrieve(q, EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, kb=KB_STRESS, top_k=top)
    return [c.source_file for c in chunks], (chunks[0].source_file if chunks else None)


async def eval_question(row: dict, top: int, sem: asyncio.Semaphore) -> dict:
    """对一题的 3 个问法做检索，返回一致性统计。"""
    questions = [row["question"], *row.get("variants", [])]
    results = []
    for q in questions:
        sources, top1 = await _retrieve_one(q, top, sem)
        rank = first_hit_rank_sources(sources, row["source"], row["answer_span"])
        results.append({"q": q, "rank": rank, "top1": top1})

    all_hit = all(r["rank"] is not None for r in results)
    top1s = {r["top1"] for r in results}
    return {"id": row["id"], "question": row["question"], "results": results,
            "all_hit": all_hit, "top1_consistent": len(top1s) == 1}


def first_hit_rank_sources(sources: list[str], expected: str, span: str) -> int | None:
    """改述评测里，hit 判定需要答案片段——但这里拿不到 chunk content。

    简化：命中 = 期望的 source 出现在 top-k 里。片段级校验见检索评测（run_eval）。
    """
    for i, s in enumerate(sources, 1):
        if s == expected:
            return i
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    rows = load_rows()
    if not rows:
        print("未找到 dataset_paraphrase.jsonl，请先运行 eval/gen_paraphrase.py")
        return
    print(f"{len(rows)} 题 × 3 问法，top={args.top}\n", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    stats = await asyncio.gather(*[eval_question(r, args.top, sem) for r in rows])

    n = len(stats)
    all_hit = sum(1 for s in stats if s["all_hit"])
    top1_consistent = sum(1 for s in stats if s["top1_consistent"])
    # 变体间命中数（0~3）
    hit_counts = [sum(1 for r in s["results"] if r["rank"] is not None) for s in stats]

    print(f"{'指标':<22}{'数值':<10}")
    print("-" * 32)
    print(f"{'全命中一致率(3问法全中)':<22}{all_hit / n:<10.3f}")
    print(f"{'Top1来源一致率':<22}{top1_consistent / n:<10.3f}")
    for k in (3, 2, 1, 0):
        c = hit_counts.count(k)
        print(f"{'命中' + str(k) + '/3 问法':<22}{c:<10}（{c / n:.1%}）")

    print("\n=== 不一致明细（3 问法结果分叉）===")
    for s in stats:
        if not s["all_hit"] or not s["top1_consistent"]:
            ranks = [r["rank"] for r in s["results"]]
            tops = [r["top1"] for r in s["results"]]
            print(f"  #{s['id']} {s['question'][:24]:<26} 命中={ranks} top1={[ (t or '—')[:18] for t in tops]}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
