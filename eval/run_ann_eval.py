"""E1：测量 HNSW 近似索引**自身**损失的召回（与语料难度无关）。

方法：同一条 query 跑两遍——
  - **近似**：走 HNSW 索引（线上实际路径）
  - **精确**：`SET LOCAL enable_indexscan = off` 强制顺序扫描 + 排序（真值）
比较两者 top-K 的**重合度**，即 ANN recall。

用途：把「检索引擎的损失」从「语料难度」中剥离出来——
之前所有召回率下降，都分不清是索引丢的还是语料难的。

用法：
    python eval/run_ann_eval.py                 # 默认 50 题、K=10
    python eval/run_ann_eval.py --n 80 --k 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.connection import close_pool, get_pool  # noqa: E402
from src.embedding.base import get_embedding  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset_clear.jsonl"
EF_SWEEP = (40, 100, 200)  # 40 是 pgvector 默认值

_SQL = """
    SELECT c.id::text AS id
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.is_deleted = false AND c.is_deprecated = false
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
"""


def _vec_str(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


async def _search(conn, vec_str: str, k: int, *, ef_search: int | None, use_index: bool) -> tuple[list[str], float]:
    """返回 (命中的 chunk id 列表, 耗时毫秒)。"""
    async with conn.transaction():
        if ef_search is not None:
            await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        if not use_index:
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
        t0 = time.perf_counter()
        rows = await conn.fetch(_SQL, vec_str, k)
        ms = (time.perf_counter() - t0) * 1000
    return [r["id"] for r in rows], ms


def _load_questions(n: int) -> list[str]:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r["question"] for r in rows[:n]]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="用多少条 query")
    parser.add_argument("--k", type=int, default=10, help="top-K")
    args = parser.parse_args()

    questions = _load_questions(args.n)
    embedding = get_embedding()
    vectors = await embedding.embed(questions)
    print(f"{len(questions)} 条 query，K={args.k}\n")

    pool = await get_pool()
    async with pool.acquire() as conn:
        # 现状信息
        total = await conn.fetchval("SELECT count(*) FROM chunks")
        ef_now = await conn.fetchval("SHOW hnsw.ef_search")
        print(f"库中切片 {total}，当前 hnsw.ef_search = {ef_now}\n")

        # 精确真值（每个 query 只算一次）
        truth: list[list[str]] = []
        exact_ms = []
        for v in vectors:
            ids, ms = await _search(conn, _vec_str(v), args.k, ef_search=None, use_index=False)
            truth.append(ids)
            exact_ms.append(ms)

        ks = [k for k in (1, 3, 5, 10) if k <= args.k]
        header = f"{'模式':<18}" + "".join(f"{'ANN@' + str(k):<10}" for k in ks) + "延迟"
        print(header)
        print("-" * len(header))
        print(f"{'精确(无索引)':<18}" + "".join(f"{'1.000':<10}" for _ in ks)
              + f"{sum(exact_ms) / len(exact_ms):.2f} ms")

        for ef in EF_SWEEP:
            per_k = {k: 0 for k in ks}
            lat = []
            for v, t in zip(vectors, truth):
                ids, ms = await _search(conn, _vec_str(v), args.k, ef_search=ef, use_index=True)
                lat.append(ms)
                for k in ks:
                    per_k[k] += len(set(ids[:k]) & set(t[:k]))
            n = len(truth)
            tag = " (默认)" if ef == 40 else ""
            cells = "".join(f"{per_k[k] / (n * k):<10.3f}" for k in ks)
            print(f"{'HNSW ef=' + str(ef) + tag:<18}{cells}{sum(lat) / len(lat):.2f} ms")

    print("\n解读：看 ANN@1 与 ANN@10 的差距——若 ANN@1 接近 1 而 ANN@10 明显低，")
    print("      说明索引的损失集中在**尾部**（低排名项），对端到端召回影响有限。")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
