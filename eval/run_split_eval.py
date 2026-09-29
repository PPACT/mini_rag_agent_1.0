"""日常集 / 鲁棒集 对照评测（同一批题、两套语料）。

**为什么这样分**：先前整个语料 726 篇里 624 篇（86%）是刻意造的近重复干扰，
而所有题都在这套语料上测 —— 等于一直测「抗干扰能力」却当成「日常表现」看，
分不清"系统不行"还是"场景太狠"。

| 测试集 | 语料 | 测什么 | 期望 |
|---|---|---|---|
| **日常集** | 干净语料（基础 16 + 通用 86，无近重复） | 常规表现 | 应很高 |
| **鲁棒集** | 完整语料（+624 篇干扰） | 抗干扰上限 | 会掉，但要**知道掉多少** |

用法（会自动清库并分别灌两套语料，耗时数分钟）：
    python eval/run_split_eval.py
    python eval/run_split_eval.py --dataset dataset_exact.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

import ingest  # noqa: E402
from run_eval import KS, load_dataset, rank_all, summarize  # noqa: E402
from src.db.connection import close_pool  # noqa: E402

CONFIGS = [
    ("日常集（干净语料）", True),
    ("鲁棒集（含干扰）", False),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_clear.jsonl")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    dataset = load_dataset(EVAL_DIR / args.dataset)
    print(f"数据集 {args.dataset}：{len(dataset)} 条；每题取回 top {args.top}")
    print("（走**生产配置**：混合检索 + 精排，改写已关）\n")

    results: dict[str, tuple[int, dict]] = {}
    for label, exclude_dept in CONFIGS:
        print(f"=== {label} ===", flush=True)
        await ingest.truncate()
        total = await ingest.ingest(exclude_dept)
        ranks = await rank_all(dataset, args.top)
        results[label] = (total, summarize(ranks))
        print(flush=True)

    header = f"{'测试集':<20}{'切片':<8}" + "".join(f"{'Recall@' + str(k):<12}" for k in KS) + "MRR"
    print(header)
    print("-" * len(header))
    for label, _ in CONFIGS:
        total, s = results[label]
        cells = "".join(f"{s['recall@' + str(k)]:<12.3f}" for k in KS)
        print(f"{label:<20}{total:<8}{cells}{s['mrr']:.3f}")

    (_, clean), (_, disturb) = (results[c[0]] for c in CONFIGS)
    print("\n【差距 = 抗干扰代价】")
    for k in KS:
        key = f"recall@{k}"
        print(f"  Recall@{k}: {clean[key]:.3f} → {disturb[key]:.3f}  ({disturb[key] - clean[key]:+.3f})")
    print(f"  MRR      : {clean['mrr']:.3f} → {disturb['mrr']:.3f}  ({disturb['mrr'] - clean['mrr']:+.3f})")

    print("\n解读：日常集分数反映「真实可用性」；两者之差反映「抗干扰代价」。")
    print("      理想状态：日常集高、差距小。若日常集就低 → 是真问题，必须修。")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
