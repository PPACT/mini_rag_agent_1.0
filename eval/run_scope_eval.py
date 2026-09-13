"""E5：验证「范围收敛」能否解决近重复带来的检索崩溃。

背景：评测语料此前**把所有文档都标成 IT**，人为制造了一个真实世界不存在的歧义
（40 个部门的同一制度全部可见）。真实场景中，用户的可见范围 = **公司级 + 本部门**。

本实验对比同一批清晰题在两种范围下的召回：
  - **无范围过滤**（现状）：全部 1231 切片都在候选池里
  - **公司级 + 本部门**（模拟 IT 用户）：候选池大幅缩小

用法（需先 python eval/ingest.py 灌入带范围标注的语料）：
    python eval/run_scope_eval.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.connection import close_pool, get_pool  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.vector_store.base import Chunk  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset_clear.jsonl"
COMPANY_SCOPE = "公司"
KS = (1, 3, 5)
CONCURRENCY = 5


def first_hit_rank(chunks: list[Chunk], span: str, source: str) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if c.source_file == source and span in c.content:
            return rank
    return None


async def rank_all(dataset: list[dict], departments, top: int, sem: asyncio.Semaphore) -> list[int | None]:
    async def one(row: dict) -> int | None:
        async with sem:
            _, chunks = await retrieve(
                row["question"], departments, 3, use_rewrite=False, use_rerank=False, top_k=top
            )
        return first_hit_rank(chunks, row["answer_span"], row["source"])

    return list(await asyncio.gather(*[one(r) for r in dataset]))


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"mrr": sum(1.0 / r for r in ranks if r) / n if n else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dept", default="IT", help="模拟用户所属部门")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = await get_pool()
    total = await pool.fetchval("SELECT count(*) FROM chunks")
    in_scope = await pool.fetchval(
        "SELECT count(*) FROM chunks WHERE department = ANY($1)", [args.dept, COMPANY_SCOPE]
    )
    print(f"库中切片 {total}；「{COMPANY_SCOPE} + {args.dept}」范围内 {in_scope} "
          f"（{in_scope / total:.1%}）\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    configs = [
        ("无范围过滤（现状）", None),
        (f"{COMPANY_SCOPE} + {args.dept}", [args.dept, COMPANY_SCOPE]),
    ]

    results = {}
    for name, depts in configs:
        results[name] = await rank_all(rows, depts, args.top, sem)

    header = f"{'范围':<22}" + "".join(f"{'Recall@' + str(k):<12}" for k in KS) + "MRR"
    print(header)
    print("-" * len(header))
    for name, _ in configs:
        s = summarize(results[name])
        print(f"{name:<22}" + "".join(f"{s['recall@' + str(k)]:<12.3f}" for k in KS) + f"{s['mrr']:.3f}")

    print("\n说明：本实验走**纯向量检索**（关改写/精排/混合）以排除 LLM 非确定性干扰，")
    print("      因此测的是「范围收敛本身」的贡献。")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
