"""检索质量评测：对比多种检索策略的 Recall@K / MRR。

**只跑「清晰题」**（唯一答案）——歧义题的"标准答案"是任意的，用它测召回会失真
（歧义识别另见 `run_clarify_eval.py`）。

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
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.vector_store.base import Chunk  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset_clear.jsonl"
# 可见范围 = 公司级 + 本部门（与线上模型一致）。
# 注意：语料按真实范围标注后，只传 ["IT"] 会把公司级文档全部过滤掉 → 召回全灭。
EVAL_DEPARTMENT = ["IT", "公司"]
EVAL_SECRET_LEVEL = 3
KS = (1, 3, 5)
CONCURRENCY = 5   # 并发题数（每题含改写/精排等 LLM 调用，串行会非常慢）

MODES: list[tuple[str, dict]] = [
    ("基线", {"use_rewrite": False, "use_rerank": False, "use_hybrid": False}),
    ("+混合", {"use_rewrite": False, "use_rerank": False, "use_hybrid": True}),
    ("改写+精排", {"use_rewrite": True, "use_rerank": True, "use_hybrid": False}),
    ("全开(生产)", {"use_rewrite": True, "use_rerank": True, "use_hybrid": True}),
    # 短链路候选：关掉改写（LLM 环节 4→3），验证是否以少量召回换稳定
    ("短链路(无改写)", {"use_rewrite": False, "use_rerank": True, "use_hybrid": True}),
]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
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
    """并发跑一批题（每题走完整检索链路，含多次 LLM 调用，串行会极慢）。"""
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(row: dict) -> int | None:
        async with sem:
            _, chunks = await retrieve(
                row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, kb=KB_STRESS, top_k=top, **kw
            )
        return first_hit_rank(chunks, row["answer_span"], row["source"])

    return list(await asyncio.gather(*[one(r) for r in dataset]))


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
    parser.add_argument("--dataset", default="dataset_clear.jsonl",
                        help="数据集文件名（默认清晰题；可用 dataset_exact.jsonl 测精确词）")
    args = parser.parse_args()

    modes = MODES
    if args.modes:
        wanted = {m.strip() for m in args.modes.split(",")}
        modes = [m for m in MODES if m[0] in wanted]
        missing = wanted - {m[0] for m in modes}
        if missing:
            print(f"未知模式: {missing}；可选: {[m[0] for m in MODES]}")
            return

    dataset = load_dataset(Path(__file__).resolve().parent / args.dataset)
    print(f"数据集 {args.dataset}：{len(dataset)} 条，每题取回 top {args.top}\n")

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
