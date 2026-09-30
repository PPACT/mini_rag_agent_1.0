"""验证 D9-⑪：向量检索到底走不走 HNSW 索引？（**修复前 / 修复后对照**）

背景：原 SQL 是

    SELECT ..., (1 - (c.embedding <=> $1::vector)) * (weight) AS score
    ... ORDER BY score DESC LIMIT n

而 pgvector 的 HNSW 索引只在 **`ORDER BY <向量列> <=> <查询向量>`** 这种
**直接用运算符**的形态下生效；`ORDER BY <表达式>` 会退化成 **Seq Scan + Sort**。

⚠️ 而且旧结论已被后来的改动架空：**E1 那次「HNSW 1.87ms vs 全表 6.83ms」是
2026-09-12 测的，软过滤的 `* weight` 表达式是 2026-09-17 才加进去的**。
1231 切片下看不出差别（全表也才几毫秒），**上量后是数量级差异**。

本脚本对照三种形态：

  A. **修复前**：表达式 + `ORDER BY score DESC`（写死的基线，已不再由代码产生）
  B. **当前代码产物**：`PgVectorStore._build_search_sql(...)`，无软过滤
     → 期望 **Index Scan**（D9-⑪ 修复点）
  C. **当前代码产物**：软过滤开启（真在降权）
     → 仍是 Seq Scan + Sort，**这是该形态的固有代价**，不是回归

用法：python eval/verify_hnsw_plan.py        （需压测库在线：docker compose --profile stress up -d）
"""
from __future__ import annotations

import asyncio
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.connection import close_pool, get_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.vector_store.base import AccessFilter  # noqa: E402
from src.vector_store.pg_vector import PgVectorStore  # noqa: E402

LIMIT = 5
REPEAT = 5  # 延迟对照的重复次数（取中位数）

# 精确定位 `chunks c` 节点的扫描方式（见 _summarize 的说明）
_CHUNK_NODE = re.compile(
    r"(?P<kind>Seq Scan|Index Scan using (?P<idx>\S+)|Index Only Scan using (?P<idx2>\S+))"
    r" on chunks c\b\s*\(cost=[^)]*\)\s*\(actual time=[\d.]+\.\.(?P<ms>[\d.]+) rows=(?P<rows>\d+)"
)

# A. 修复前形态：给距离乘 1.0 再按表达式排序 —— HNSW 认不出
FORM_A = """
SELECT c.id::text, (1 - (c.embedding <=> $1::vector)) * (1.0) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.is_deleted = false AND c.is_deprecated = false
ORDER BY score DESC
LIMIT {k}
"""


def _summarize(plan_rows: list) -> dict:
    """从 EXPLAIN 输出里抽出 **chunks 那个节点** 的信息。

    ⚠️ 必须**只看 `chunks c` 节点**：计划里 `documents d` 也会出现 Seq Scan /
    Index Scan，按整段文本 grep 会把「文档表的扫描方式」当成「向量检索的方式」——
    结论可能因此恰好正确、也可能静默错反。故这里定位到具体节点。

    ⚠️ 节点行形如 `Seq Scan on chunks c  (cost=...) (actual time=... rows=N)`：
    `(cost=...)` 夹在中间，正则必须显式跳过它，否则匹配失败后
    `rows_scanned` 会**静默取默认值 0**（这个坑第一次写就踩了）。
    """
    txt = "\n".join(r["QUERY PLAN"] for r in plan_rows)
    info = {
        "found": False,
        "seq_scan": False,
        "index_scan": False,
        "index_name": None,
        "sort": "Sort" in txt,
        "rows_scanned": 0,
        "node_ms": 0.0,
        "exec_ms": 0.0,
    }
    m = _CHUNK_NODE.search(txt)
    if m:
        kind = m.group("kind")
        info["found"] = True
        info["seq_scan"] = kind == "Seq Scan"
        info["index_scan"] = not info["seq_scan"]
        info["index_name"] = m.group("idx") or m.group("idx2")
        info["rows_scanned"] = int(m.group("rows"))
        info["node_ms"] = float(m.group("ms"))
    m2 = re.search(r"Execution Time: ([\d.]+) ms", txt)
    if m2:
        info["exec_ms"] = float(m2.group(1))
    return info


async def _plan(pool, sql: str, params: list) -> dict:
    rows = await pool.fetch("EXPLAIN (ANALYZE, BUFFERS)\n" + sql, *params)
    return _summarize(rows)


async def _latency(pool, sql: str, params: list) -> tuple[float, float]:
    """真实执行 REPEAT 次，返回 (中位数, 最大值) 毫秒。

    先跑一次不计时（预热缓存），否则第一次会把冷启动开销算进来——
    那正是 A/B 之间最容易被误读成"谁更快"的地方。
    """
    await pool.fetch(sql, *params)
    samples = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        await pool.fetch(sql, *params)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), max(samples)


def _shape(info: dict) -> str:
    if not info["found"]:
        return "⚠️ 未识别到 chunks 节点（判读不可信）"
    if info["index_scan"]:
        return f"Index Scan ✅ 用上 HNSW（{info['index_name']}）"
    return "Seq Scan ❌ 索引未使用" + (" + Sort" if info["sort"] else "")


async def main() -> None:
    pool = await get_pool(KB_STRESS)
    vec = await pool.fetchval("SELECT embedding::text FROM chunks LIMIT 1")
    total = await pool.fetchval("SELECT count(*) FROM chunks")
    print(f"语料：{total} 切片（压测库）  ·  每种形态重复 {REPEAT} 次取中位数\n")

    store = PgVectorStore(KB_STRESS)

    # B = 当前代码在「无软过滤」下的真实产物（D9-⑪ 修改点）
    sql_b, params_b = store._build_search_sql(vec, AccessFilter(), LIMIT)
    # C = 当前代码在「软过滤开启」下的产物
    sql_c, params_c = store._build_search_sql(
        vec, AccessFilter(departments=["IT"]), LIMIT, soft_scope=True)

    cases = [
        ("A. 修复前：ORDER BY score DESC（表达式）", FORM_A.format(k=LIMIT), [vec]),
        ("B. 修复后：代码产物（无软过滤）", sql_b, params_b),
        ("C. 修复后：代码产物（软过滤开启）", sql_c, params_c),
    ]

    print("| 形态 | chunks 节点计划 | 该节点扫描行数 | EXPLAIN 总 ms | 实测中位 ms | 实测最大 ms |")
    print("|---|---|---|---|---|---|")
    for name, sql, params in cases:
        info = await _plan(pool, sql, params)
        med, mx = await _latency(pool, sql, params)
        print(f"| {name} | {_shape(info)} | {info['rows_scanned']} | "
              f"{info['exec_ms']:.2f} | {med:.2f} | {mx:.2f} |")

    print("\n判读：")
    print("  · B 的 chunks 节点是 **Index Scan** → D9-⑪ 修复生效（修复前 A 为 Seq Scan）")
    print("  · **扫描行数**是最硬的指标：A 扫全表、B 只扫 LIMIT 行 → 差距随语料线性放大。")
    print("  · C 仍是 Seq Scan 属**预期**：降权排序只能用表达式，索引用不上。")
    print("  · ⚠️ 修复后 `hnsw.ef_search` 才真正生效（修复前是空转），")
    print("    E1「ef_search 40→100 恢复满召回」届时重新适用（见 优化方案 §5 R11）。")
    print("  · ⚠️ **本脚本只回答「走不走索引」，不回答「结果变了多少」** ——")
    print("    走索引 = 近似检索，top-K 可能与精确扫描不同。召回对照见 eval/run_ann_eval.py。")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
