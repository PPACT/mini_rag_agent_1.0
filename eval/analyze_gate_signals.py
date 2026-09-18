"""测量「门控信号」的区分度：哪个信号能分开「歧义」与「不歧义」？

背景：歧义判定占 56.7% 延迟，但 97.5% 的结果是「不歧义」——需要门控砍掉多数调用。
上一次门控（"来源分散度 ≥3"）是**拍脑袋定的阈值**，通过率 98%，失败。
**本次先测量后设计**：对两组题计算候选信号，看哪个能分开。

三组对照：
  - 不歧义组：dataset_clear.jsonl（80 题）
  - 歧义组  ：dataset_clarify.jsonl 里 kind=ambiguous（25 题）
  - 对抗组  ：dataset_clarify.jsonl 里 kind=adversarial（12 题）

三个候选信号（都基于**精排后的 top-K**，与线上链路一致）：
  A 来源分散度：top-K 来自多少个不同文档      （上次失败的信号）
  B 分数平坦度：top1.score - topK.score      （gap 小 = 无赢家）
  C 候选互似度：top-K 彼此的平均余弦相似度     （高 = 近重复）

用法：python eval/analyze_gate_signals.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from src.db.connection import close_pool, get_pool  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

DEPTS = ["IT", "公司"]
LEVEL = 3
TOP_K = 5
CONCURRENCY = 5


def _cos(a: str, b: str) -> float:
    """解析 pgvector 文本格式 '[0.1,0.2,...]' 并算余弦相似度。"""
    va = [float(x) for x in a.strip("[]").split(",")]
    vb = [float(x) for x in b.strip("[]").split(",")]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    return dot / (na * nb + 1e-9)


async def _signals(question: str, sem: asyncio.Semaphore, pool) -> dict | None:
    async with sem:
        _, chunks = await retrieve(question, DEPTS, LEVEL, use_rerank=True, top_k=TOP_K)
    if len(chunks) < 2:
        return None
    ids = [c.id for c in chunks if c.id]
    rows = await pool.fetch(
        "SELECT id::text, embedding::text AS emb FROM chunks WHERE id = ANY($1::uuid[])", ids)
    embs = {r["id"]: r["emb"] for r in rows}

    sims = [
        _cos(embs[a], embs[b])
        for i, a in enumerate(ids) for b in ids[i + 1:]
        if a in embs and b in embs
    ]
    return {
        "sources": len({c.source_file for c in chunks}),
        "flatness": chunks[0].score - chunks[-1].score,
        "avg_sim": sum(sims) / len(sims) if sims else 0.0,
    }


async def _run_group(name: str, questions: list[str], sem, pool) -> list[dict]:
    res = await asyncio.gather(*[_signals(q, sem, pool) for q in questions])
    out = [r for r in res if r]
    print(f"  {name}: {len(out)} 题", flush=True)
    return out


def _stat(group: list[dict], key: str) -> tuple[float, float]:
    vals = [g[key] for g in group]
    return (sum(vals) / len(vals) if vals else 0, max(vals) if vals else 0)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每组取样上限（0=全部）")
    args = ap.parse_args()

    clear = [json.loads(l) for l in (EVAL_DIR / "dataset_clear.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    clar = [json.loads(l) for l in (EVAL_DIR / "dataset_clarify.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups_q = {
        "不歧义组": [r["question"] for r in clear],
        "歧义组": [r["question"] for r in clar if r.get("kind") == "ambiguous"],
        "对抗组": [r["question"] for r in clar if r.get("kind") == "adversarial"],
    }
    if args.limit:
        groups_q = {k: v[: args.limit] for k, v in groups_q.items()}

    print("计算三组信号（生产配置：开精排）...", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    pool = await get_pool()
    groups = {k: await _run_group(k, v, sem, pool) for k, v in groups_q.items()}

    print(f"\n{'信号':<14}{'不歧义组':<16}{'歧义组':<16}{'对抗组':<16}{'区分度'}")
    print("-" * 76)
    for key, label, higher_means_ambig in (
        ("sources", "来源分散度", True),
        ("flatness", "分数平坦度(gap)", False),
        ("avg_sim", "候选互似度", True),
    ):
        cells = []
        means = {}
        for gname, g in groups.items():
            m, mx = _stat(g, key)
            means[gname] = m
            cells.append(f"{m:.3f}(max {mx:.2f})")
        # 区分度：歧义(含对抗)组 与 不歧义组 的均值差
        amb = (means["歧义组"] + means["对抗组"]) / 2
        diff = abs(amb - means["不歧义组"])
        verdict = "✅ 有区分" if diff > 0.05 else "❌ 无区分"
        print(f"{label:<14}{cells[0]:<16}{cells[1]:<16}{cells[2]:<16}{verdict} (Δ={diff:.3f})")

    print("\n解读：Δ 越大越能分开两组 → 用该信号做门控。三个都没区分度 → 门控不可行。")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
