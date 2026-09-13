"""澄清行为评测：测歧义识别的「漏报率」与「误报率」。

- **歧义题 / 对抗题** → 系统**应该**触发澄清；没触发 = 漏报
- **清晰题** → 系统**不应**触发澄清；触发了 = 误报（过度打扰）

用法（需先 ingest 语料）：
    python eval/run_clarify_eval.py
    python eval/run_clarify_eval.py --top 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.connection import close_pool  # noqa: E402
from src.rag.ambiguity import check_ambiguity, is_diverse  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

EVAL = Path(__file__).resolve().parent
CLARIFY_SET = EVAL / "dataset_clarify.jsonl"
CLEAR_SET = EVAL / "dataset_clear.jsonl"
EVAL_DEPARTMENT = ["IT"]
EVAL_SECRET_LEVEL = 3


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _one(row: dict, top: int, sem: asyncio.Semaphore) -> tuple[dict, bool, bool]:
    settings = get_settings()
    async with sem:
        _, chunks = await retrieve(row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL, top_k=top)
        gated = is_diverse(chunks, settings.ambiguity_source_threshold)
        result = await check_ambiguity(row["question"], chunks)
        return (row, gated, result.ambiguous)


async def run_set(rows: list[dict], top: int, sem: asyncio.Semaphore) -> list[tuple[dict, bool, bool]]:
    """并发跑一批题；返回 [(题目, 是否门控通过, 是否判定歧义)]。

    并发是必需的：每题要走完整生产链路（改写 + 精排 + 歧义判定 ≈ 3-4 次 LLM 调用），
    串行跑 117 题需要半小时以上。
    """
    return list(await asyncio.gather(*[_one(r, top, sem) for r in rows]))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="检索取回条数（与线上一致）")
    parser.add_argument("--concurrency", type=int, default=5, help="并发题数")
    args = parser.parse_args()
    sem = asyncio.Semaphore(args.concurrency)

    clarify_rows = _load(CLARIFY_SET)
    clear_rows = _load(CLEAR_SET)
    if not clarify_rows and not clear_rows:
        print("评测集为空，请先运行 eval/gen_questions.py")
        return

    print(f"歧义/对抗题 {len(clarify_rows)} 条，清晰题 {len(clear_rows)} 条，top={args.top}\n")

    # 应澄清组
    print("跑「应澄清」组...", flush=True)
    should = await run_set(clarify_rows, args.top, sem)
    # 不应澄清组
    print("跑「不应澄清」组...", flush=True)
    should_not = await run_set(clear_rows, args.top, sem)

    amb = [(r, g, a) for r, g, a in should if r.get("kind") == "ambiguous"]
    adv = [(r, g, a) for r, g, a in should if r.get("kind") == "adversarial"]

    print(f"{'类别':<14}{'题数':<8}{'触发澄清':<10}{'漏报率':<10}")
    print("-" * 44)
    for name, items in (("歧义题", amb), ("对抗题", adv)):
        if not items:
            continue
        hit = sum(1 for _, _, a in items if a)
        print(f"{name:<14}{len(items):<8}{hit:<10}{1 - hit / len(items):<10.1%}")
    if should:
        hit = sum(1 for _, _, a in should if a)
        print(f"{'合计(应澄清)':<14}{len(should):<8}{hit:<10}{1 - hit / len(should):<10.1%}")

    if should_not:
        fp = sum(1 for _, _, a in should_not if a)
        print(f"\n{'清晰题(不应澄清)':<18}{len(should_not)} 条，误报 {fp} 条 → 误报率 {fp / len(should_not):.1%}")

    # 门控统计（诊断用）
    gated_yes = sum(1 for _, g, _ in should + should_not if g)
    print(f"\n门控通过率: {gated_yes}/{len(should + should_not)}（未通过则直接跳过 LLM 判定）")

    print("\n=== 漏报明细（应澄清但没澄清）===")
    missed = [(r, g) for r, g, a in should if not a]
    for r, g in missed[:15]:
        tag = "门控拦截" if not g else "判定为非歧义"
        print(f"  #{r['id']} [{r.get('kind')}] {r['question'][:26]:<28} ← {tag}")
    if not missed:
        print("  无")

    print("\n=== 误报明细（不该澄清却澄清）===")
    fps = [r for r, _, a in should_not if a]
    for r in fps[:10]:
        print(f"  #{r['id']} {r['question'][:30]}")
    if not fps:
        print("  无")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
