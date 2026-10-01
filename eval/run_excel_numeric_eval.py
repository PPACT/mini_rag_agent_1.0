"""`2.0-47` —— Excel 表格块**偏长**会不会影响召回？**只出数字，不改参数**。

## 为什么（`交流区 §1.37` 判定 / `§1.43` 出题）

`_XLSX_ROWS_PER_BLOCK = 60` 被怀疑对**宽表**太大 → 文档侧裁定 **先量再改**：
「**R@K 好 → 『偏长』不是问题，本项关闭；R@K 差 → 再决定**」。
⚠️ **不许跳过 L1 直接调参**（`2.0-39` 的门槛）。

## 三项输出（`§1.43⑤`）

| # | 量什么 | 来源 |
|---|---|---|
| ① | Excel 表格块块长分布 p50 / p90 / max | 纯 CPU |
| ② | 超长块占比（> 2 × chunk_size） | 纯 CPU |
| ③ | **R@1 / R@3 / R@5** | 需已灌库 |

## ⚠️ 评分口径（`§1.43③`，**先定死再看数**）

- **看召回** —— 「出处」那一块**在不在 top-K**；**⛔ 不是看生成的答案对不对**。
- **命中** = 来自**期望文件** 且 该块**包含这道题的全部 `answer_spans`**
  —— 不是"文件对上了就算"。
- K 取 **1 / 3 / 5 都报**。

## ⭐ 台子自检（**必须有，否则 0 分是假的**）

`answer_spans` 是**逐字**匹配 chunk 文本的。渲染一改，span 就失配 →
**召回全 0，而这个 0 会被当成"块偏长导致召回差"的证据**。
所以开跑前先做一遍**本地核对**（不需要 DB）：每题的 span 必须全都能在
**期望文件的某个 chunk** 里找到；**任一题不满足就终止**，不产出数字。

用法（需先 `python eval/ingest.py` 灌语料 + Docker/Ollama 在跑）：
    python eval/run_excel_numeric_eval.py
    python eval/run_excel_numeric_eval.py --modes 全开(生产) --top 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.document_parser.chunking import split_blocks  # noqa: E402
from src.document_parser.routing import parse_document  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.vector_store.base import Chunk  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset_excel_numeric.jsonl"
CORPUS = Path(__file__).resolve().parents[1] / "docs" / "corpus"
EVAL_DEPARTMENT = ["IT", "公司"]
EVAL_SECRET_LEVEL = 3
KS = (1, 3, 5)
CONCURRENCY = 5

XLSX = ("06_供应商准入与评审标准", "11_客户服务响应标准SLA", "12_固定资产管理办法",
        "16_2025年度培训计划", "18_IT资产台账_2025Q3", "20_公司简介与组织架构",
        "21_费用预算明细_2025")

MODES: list[tuple[str, dict]] = [
    ("基线", {"use_rewrite": False, "use_rerank": False, "use_hybrid": False}),
    ("+混合", {"use_rewrite": False, "use_rerank": False, "use_hybrid": True}),
    ("改写+精排", {"use_rewrite": True, "use_rerank": True, "use_hybrid": False}),
    ("全开(生产)", {"use_rewrite": True, "use_rerank": True, "use_hybrid": True}),
]
DEFAULT_MODES = "基线"


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _pct_nearest_rank(vals: list[int], q: int) -> int:
    """**nearest-rank** 分位。

    ⚠️ **别用 `statistics.quantiles` 的默认 `exclusive` 方法** —— 小样本上它会**外推**，
    实测在 2 个值的样本上算出 `p99 > max`（Excel 表格块常只有 2~4 个，正是小样本）。
    """
    v = sorted(vals)
    if not v:
        return 0
    return v[min(len(v) - 1, max(0, math.ceil(q / 100 * len(v)) - 1))]


def block_lengths() -> tuple[list[dict], list[int]]:
    """①② 逐文件块长（**只用表格块**）+ 汇总原始长度。纯 CPU，不需要 DB。"""
    s = get_settings()
    rows, allv = [], []
    for name in XLSX:
        doc = parse_document(str(CORPUS / f"{name}.xlsx"))
        lens = [len(c.text) for c in split_blocks(doc, s.chunk_size, s.chunk_overlap)
                if c.kind == "table"]
        allv += lens
        rows.append({"file": name, "n": len(lens), "max": max(lens) if lens else 0,
                     "p50": _pct_nearest_rank(lens, 50), "p90": _pct_nearest_rank(lens, 90),
                     "over": sum(1 for x in lens if x > 2 * s.chunk_size), "lens": sorted(lens)})
    return rows, allv


def selfcheck_spans(dataset: list[dict]) -> list[str]:
    """⭐ 开跑前的本地核对：每题的 span 必须**全部**落在期望文件的某个 chunk 里。

    ⚠️ 不做这一步，渲染一变就会**静默全 0**，而那个 0 会被读成"块偏长导致召回差"。
    返回**不合格题目**的说明（空 = 全部通过）。
    """
    s = get_settings()
    cache: dict[str, list[str]] = {}
    bad: list[str] = []
    for row in dataset:
        src = row["source"]
        if src not in cache:
            doc = parse_document(str(CORPUS / src))
            cache[src] = [c.text for c in split_blocks(doc, s.chunk_size, s.chunk_overlap)]
        texts = cache[src]
        missing = [sp for sp in row["answer_spans"]
                   if not any(sp in t for t in texts)]
        if missing:
            bad.append(f"#{row['id']} {row['question'][:20]} → 本地 chunk 里找不到：{missing}")
    return bad


def hit_rank(chunks: list[Chunk], spans: list[str], source: str) -> int | None:
    """首个命中的切片排名（1-based）；未命中 → None。

    命中 = 来自**期望文件** 且 **全部** span 都在 `content` 里（`§1.43③` 的口径）。
    """
    for rank, c in enumerate(chunks, start=1):
        if c.source_file == source and all(sp in (c.content or "") for sp in spans):
            return rank
    return None


async def rank_all(dataset: list[dict], top: int, **kw) -> list[int | None]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(row: dict) -> int | None:
        async with sem:
            _, chunks = await retrieve(
                row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL,
                kb=KB_STRESS, top_k=top, **kw)
        return hit_rank(chunks, row["answer_spans"], row["source"])

    return list(await asyncio.gather(*[one(r) for r in dataset]))


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n, "mrr": sum(1.0 / r for r in ranks if r) / n if n else 0.0}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n if n else 0.0
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="每次检索取回的切片数")
    ap.add_argument("--modes", default=DEFAULT_MODES,
                    help=f"逗号分隔，默认 {DEFAULT_MODES}（另可选：改写+精排 / 全开(生产)）")
    ap.add_argument("--no-write", action="store_true", help="不写 JSON 产物")
    args = ap.parse_args()

    s = get_settings()
    dataset = load_dataset(DATASET)
    print(f"2.0-47 Excel 数值类迷你集 ｜ {len(dataset)} 题 ｜ chunk_size={s.chunk_size} "
          f"overlap={s.chunk_overlap} ｜ 超长阈值 >{2 * s.chunk_size}\n")

    # ---- ⭐ 口径自检（无 DB 也能跑）----
    bad = selfcheck_spans(dataset)
    if bad:
        print("[X] **口径自检没过 —— 不产出任何召回数字**（否则 0 分是假的）：")
        for b in bad:
            print(f"    · {b}")
        print("\n    修法：`answer_spans` 必须与**实际 chunk 文本**逐字一致（渲染改过就要跟着改）。")
        return
    print(f"✅ 口径自检通过：{len(dataset)}/{len(dataset)} 题的 span 都能在期望文件的 chunk 里找到\n")

    # ---- ①② 块长 ----
    rows, allv = block_lengths()
    print("【① Excel 表格块块长 ／ ② 超长块占比】（**nearest-rank 分位，不外推**）")
    print(f"  {'文件':<24}{'表块':>5}{'p50':>7}{'p90':>7}{'max':>7}{'超长':>6}   原始长度")
    for r in rows:
        print(f"  {r['file'][:22]:<24}{r['n']:>5}{r['p50']:>7}{r['p90']:>7}{r['max']:>7}"
              f"{r['over']:>6}   {r['lens']}")
    over = sum(1 for x in allv if x > 2 * s.chunk_size)
    print(f"  {'【7 篇合计】':<24}{len(allv):>5}{_pct_nearest_rank(allv, 50):>7}"
          f"{_pct_nearest_rank(allv, 90):>7}{max(allv):>7}{over:>6}")
    print(f"  超长块占比：{over}/{len(allv)} = {over / len(allv) * 100:.1f}%"
          f"  （⚠️ 本表块数很少，分位数本身不稳，**别过度解读 p90/p99**）\n")

    # ---- ③ R@K ----
    wanted = {m.strip() for m in args.modes.split(",")}
    modes = [m for m in MODES if m[0] in wanted]
    if not modes:
        print(f"[X] 未知模式 {wanted}；可选：{[m[0] for m in MODES]}")
        return

    print(f"【③ 召回】每题取回 top {args.top} ｜ 命中 = 同一文件 + **全部 span 都在该块里**")
    results: dict[str, list[int | None]] = {}
    for name, kw in modes:
        results[name] = await rank_all(dataset, args.top, **kw)

    header = f"  {'模式':<12}" + "".join(f"{'R@' + str(k):<9}" for k in KS) + "MRR"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, _ in modes:
        st = summarize(results[name])
        print(f"  {name:<12}" + "".join(f"{st['recall@' + str(k)]:<9.3f}" for k in KS)
              + f"{st['mrr']:.3f}")

    print("\n  逐题明细（排名；— = 未命中）：")
    for i, row in enumerate(dataset):
        parts = " | ".join(
            f"{name}: {'—' if results[name][i] is None else results[name][i]}"
            for name, _ in modes)
        print(f"    #{row['id']} {row['question'][:26]:<28} {parts}")

    if not args.no_write:
        out = Path(__file__).resolve().parents[1] / "logs" / "excel_numeric_eval.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "params": {"chunk_size": s.chunk_size, "overlap": s.chunk_overlap, "top": args.top},
            "blocks": rows, "blocks_total": {"n": len(allv), "over": over,
                                             "p50": _pct_nearest_rank(allv, 50),
                                             "p90": _pct_nearest_rank(allv, 90),
                                             "max": max(allv) if allv else 0},
            "modes": {name: {"summary": summarize(results[name]),
                             "ranks": results[name]} for name, _ in modes},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n产物 → logs/excel_numeric_eval.json")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
