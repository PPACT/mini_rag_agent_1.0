"""从「无变体」文档自动生成清晰题（问题 + 逐字答案片段），并校验片段真实存在。

用途：扩充评测集到 100+ 条。这些文档（eval/corpus_synth/llm*.md）在语料中唯一，
所以它们的答案**不受"多版本"影响**，适合作为「清晰题」——
用来测「检索质量」和「不该澄清时是否误澄清」。

用法：python eval/gen_questions.py
产物：eval/dataset_clear.jsonl
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.litellm_client import complete  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_synth"
OUT = Path(__file__).resolve().parent / "dataset_clear.jsonl"

_SYSTEM = """你是问答数据集构造助手。给定一篇企业制度文档，请出 1 个该文档能**唯一回答**的问题。
要求：
1. 问题要像真实员工会问的，可以口语化；
2. 同时给出答案在原文中的**逐字片段**（必须是原文里一字不差的连续子串）；
3. 片段长度 10~40 字，且包含关键信息（数字、时限、金额等）；
4. 只输出 JSON，不要任何其他文字：{"question": "...", "answer_span": "..."}"""

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


async def _gen_one(path: Path, idx: int, sem: asyncio.Semaphore) -> dict | None:
    text = path.read_text(encoding="utf-8")
    async with sem:
        try:
            raw = await complete(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text[:2500]},
                ]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] {path.name}: {e}")
            return None

    match = _JSON_OBJ.search(raw or "")
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    question = str(data.get("question", "")).strip()
    span = str(data.get("answer_span", "")).strip()
    # 关键校验：答案片段必须逐字存在于原文（否则该题无意义）
    if not question or not span or span not in text:
        return None
    return {
        "id": 1000 + idx,
        "kind": "clear",
        "question": question,
        "source": path.name,
        "answer_span": span,
        "type": "generated",
    }


async def main() -> None:
    files = sorted(CORPUS_DIR.glob("llm*.md"))
    print(f"从 {len(files)} 篇无变体文档生成清晰题（并发 5）...")
    sem = asyncio.Semaphore(5)
    results = await asyncio.gather(*[_gen_one(p, i, sem) for i, p in enumerate(files)])

    rows = [r for r in results if r]
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"生成有效清晰题 {len(rows)}/{len(files)}（片段校验未通过或生成失败的已丢弃）")
    print(f"→ {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
