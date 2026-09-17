"""为清晰题生成改述变体（语义等价但表述不同），用于测「换个问法是否稳定」。

背景：用户原始疑虑是"换个问法就崩"，但我们每题只有一个问法，从未验证。
这里为每道清晰题生成 2 个改写变体（共 3 个问法），供一致性评测使用。

用法（需 LLM）：
    python eval/gen_paraphrase.py
产物：eval/dataset_paraphrase.jsonl
    每条: {id, question, source, answer_span, variants: [改述1, 改述2]}
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.litellm_client import complete  # noqa: E402

SRC = Path(__file__).resolve().parent / "dataset_clear.jsonl"
OUT = Path(__file__).resolve().parent / "dataset_paraphrase.jsonl"

_SYSTEM = """你是企业问答的改述助手。把给定的用户问题改写成 2 个**语义完全相同**但**表述不同**的变体。
要求：
1. 保持原意、关键实体、数字语义完全一致，不改变任何约束；
2. 变体之间也要尽量不同（换句式、换措辞、口语化/书面化交替）；
3. 只输出 JSON 数组，例如 ["变体一","变体二"]，不要任何其他文字。"""

_JSON_ARR = re.compile(r"\[.*\]", re.DOTALL)


async def _gen_variants(question: str, sem: asyncio.Semaphore) -> list[str]:
    async with sem:
        try:
            raw = await complete(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": question},
                ]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] {question[:20]}: {e}")
            return []
    m = _JSON_ARR.search(raw or "")
    if not m:
        return []
    try:
        arr = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and x.strip()][:2]


async def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"为 {len(rows)} 道清晰题生成改述变体（每题 2 个）...")
    sem = asyncio.Semaphore(5)
    ok, skipped = 0, 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            variants = await _gen_variants(r["question"], sem)
            if not variants:
                skipped += 1
                continue
            f.write(json.dumps({
                "id": r["id"], "question": r["question"], "source": r["source"],
                "answer_span": r["answer_span"], "variants": variants,
            }, ensure_ascii=False) + "\n")
            ok += 1
            if (i + 1) % 20 == 0:
                print(f"  进度 {i + 1}/{len(rows)}")
    print(f"完成：{ok} 题（{skipped} 题改述失败跳过）→ {OUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
