"""检索质量评测：对比「基线」与「多查询扩展」的 Recall@K / MRR。

用法（需先 python eval/ingest.py 灌语料）：
    python eval/run_eval.py                 # 默认取 top 10 观察完整排序
    python eval/run_eval.py --top 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.connection import close_pool  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.vector_store.base import Chunk  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
EVAL_DEPARTMENT = ["IT"]
EVAL_SECRET_LEVEL = 3
KS = (1, 3, 5)


def load_dataset() -> list[dict]:
    rows = []
    with DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_hit_rank(chunks: list[Chunk], answer_span: str) -> int | None:
    """返回首个包含答案片段的切片排名（1-based）；未命中返回 None。"""
    for rank, c in enumerate(chunks, start=1):
        if answer_span in c.content:
            return rank
    return None


async def rank_all(dataset: list[dict], use_rewrite: bool, top: int) -> list[int | None]:
    ranks = []
    for row in dataset:
        _, chunks = await retrieve(
            row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL,
            use_rewrite=use_rewrite, top_k=top,
        )
        ranks.append(first_hit_rank(chunks, row["answer_span"]))
    return ranks


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n, "mrr": sum(1.0 / r for r in ranks if r) / n if n else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="每次检索取回的切片数（观察完整排序）")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"评测集 {len(dataset)} 条，每题取回 top {args.top}\n")

    base_ranks = await rank_all(dataset, False, args.top)
    rewrite_ranks = await rank_all(dataset, True, args.top)

    header = f"{'模式':<14}" + "".join(f"{'Recall@' + str(k):<12}" for k in KS) + "MRR"
    print(header)
    print("-" * len(header))
    for name, ranks in (("基线（无改写）", base_ranks), ("多查询扩展", rewrite_ranks)):
        s = summarize(ranks)
        line = f"{name:<14}" + "".join(f"{s['recall@' + str(k)]:<12.3f}" for k in KS) + f"{s['mrr']:.3f}"
        print(line)

    print("\n逐题对比（排名，越小越好；— 表示未命中）：")
    improved, worsened = [], []
    for row, b, r in zip(dataset, base_ranks, rewrite_ranks):
        bs, rs = ("—" if b is None else str(b)), ("—" if r is None else str(r))
        flag = ""
        if b and r and r < b:
            flag, _ = "  ↑ 提升", improved.append(row["id"])
        elif b and r and r > b:
            flag, _ = "  ↓ 变差", worsened.append(row["id"])
        elif b is None and r is not None:
            flag, _ = "  ↑ 由未命中变命中", improved.append(row["id"])
        elif b is not None and r is None:
            flag, _ = "  ↓ 由命中变未命中", worsened.append(row["id"])
        print(f"  #{row['id']:<3} [{row['type']:<12}] 基线 {bs:<3} -> 改写 {rs:<3}{flag}  {row['question']}")

    print(f"\n提升题号: {improved or '无'}    变差题号: {worsened or '无'}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
