"""检索质量评测：对比多种检索策略的 Recall@K / MRR。

用法（需先 python eval/ingest.py 灌语料）：
    python eval/run_eval.py                 # 跑全部模式
    python eval/run_eval.py --top 10        # 每次取回的切片数
    python eval/run_eval.py --modes 基线,改写+精排
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

MODES: list[tuple[str, dict]] = [
    ("基线", {"use_rewrite": False, "use_rerank": False}),
    ("仅改写", {"use_rewrite": True, "use_rerank": False}),
    ("仅精排", {"use_rewrite": False, "use_rerank": True}),
    ("改写+精排", {"use_rewrite": True, "use_rerank": True}),
]


def load_dataset() -> list[dict]:
    rows = []
    with DATASET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_hit_rank(chunks: list[Chunk], answer_span: str, source_file: str) -> int | None:
    """返回首个命中的切片排名（1-based）；未命中返回 None。

    命中 = 来自**期望的文档** 且 内容包含答案片段。
    校验 source_file 是必要的：语料里有同主题的其他文档（部门变体、相近主题），
    只看内容会把"别人的版本"误判为命中。
    """
    for rank, c in enumerate(chunks, start=1):
        if c.source_file == source_file and answer_span in c.content:
            return rank
    return None


async def rank_all(dataset: list[dict], top: int, **kw) -> list[int | None]:
    ranks = []
    for row in dataset:
        _, chunks = await retrieve(
            row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, top_k=top, **kw
        )
        ranks.append(first_hit_rank(chunks, row["answer_span"], row["source"]))
    return ranks


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n, "mrr": sum(1.0 / r for r in ranks if r) / n if n else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="每次检索取回的切片数")
    parser.add_argument("--modes", default=None, help="逗号分隔的模式名，默认全部")
    args = parser.parse_args()

    modes = MODES
    if args.modes:
        wanted = {m.strip() for m in args.modes.split(",")}
        modes = [m for m in MODES if m[0] in wanted]
        missing = wanted - {m[0] for m in modes}
        if missing:
            print(f"未知模式: {missing}；可选: {[m[0] for m in MODES]}")
            return

    dataset = load_dataset()
    print(f"评测集 {len(dataset)} 条，每题取回 top {args.top}\n")

    results: dict[str, list[int | None]] = {}
    for name, kw in modes:
        results[name] = await rank_all(dataset, args.top, **kw)

    header = f"{'模式':<12}" + "".join(f"{'Recall@' + str(k):<12}" for k in KS) + "MRR"
    print(header)
    print("-" * len(header))
    for name, _ in modes:
        s = summarize(results[name])
        line = f"{name:<12}" + "".join(f"{s['recall@' + str(k)]:<12.3f}" for k in KS) + f"{s['mrr']:.3f}"
        print(line)

    # 逐题对比：以第一个模式为基线
    base_name = modes[0][0]
    base_ranks = results[base_name]
    print(f"\n逐题对比（基线 = {base_name}）：")
    for i, row in enumerate(dataset):
        parts = [f"基线 {'—' if base_ranks[i] is None else base_ranks[i]}"]
        for name, _ in modes[1:]:
            r = results[name][i]
            parts.append(f"{name} {'—' if r is None else r}")
        print(f"  #{row['id']:<3} {row['question'][:22]:<24} " + " | ".join(parts))

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
