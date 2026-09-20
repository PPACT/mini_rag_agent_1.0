"""查清「10 秒长尾」的成因：是**输出太长**，还是**调用真的卡住**？

已知事实：
  - 176 次歧义判定调用里，38 次 >6s，**连续散布到 10.8s 后戛然而止**（10-11s 桶堆了 14 个，
    再往上一个都没有）——硬边界，不像随机抖动。
  - 对照实验（同一 prompt 调 40 次）**一次都没超 6s**，最大 3.2s。
  → 差别在于：测量时每题 prompt 都不同；对照组 prompt 完全一样。

**假设**：`complete()` 里 `max_tokens=2048`，模型有时会啰嗦地生成大段理由，
   生成耗时可达 ~10s；prompt 一变（snippet 档位不同）输出长度就变 → 解释了"各档慢的不是同一批题"。

**验证**：挑 300 字档最慢的 8 题 + 最快的 4 题做对照，各调两次：
  - `max_tokens=2048`（现状）→ 记录耗时 + 输出字符数
  - `max_tokens=256`（够放判定 JSON）→ 记录耗时 + 输出字符数
若「耗时与输出长度强相关」且「限长后长尾消失」→ 假设成立，对策就是把判定调用的 max_tokens 压小。

用法：python eval/measure_slowcall_cause.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402

from src.config.prompts import load_ambiguity_check_templates  # noqa: E402
from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.ambiguity import _format_candidates  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

DETAIL = ROOT / "logs" / "snippet_measure.json"
N_SLOW, N_FAST = 8, 4
DEPTS = ["IT", "公司"]
LEVEL = 3
SNIPPET = 300


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _call(msgs: list[dict], max_tokens: int, s) -> tuple[float, int]:
    t0 = time.perf_counter()
    resp = await litellm.acompletion(
        model=f"deepseek/{s.deepseek_model}", messages=msgs, api_key=s.deepseek_api_key,
        api_base=s.deepseek_base_url, temperature=0, max_tokens=max_tokens, timeout=60,
    )
    ms = (time.perf_counter() - t0) * 1000
    return ms, len(resp.choices[0].message.content or "")


async def main() -> None:
    s = get_settings()
    detail = json.loads(DETAIL.read_text(encoding="utf-8"))["detail"]["300"]
    by_ms = sorted(detail.items(), key=lambda kv: -kv[1]["ms"])
    picked = [(k, v["ms"], "慢") for k, v in by_ms[:N_SLOW]] + \
             [(k, v["ms"], "快") for k, v in by_ms[-N_FAST:]]

    # id → question
    qmap = {str(r["id"]): r["question"]
            for r in _load("dataset_clear.jsonl") + _load("dataset_clarify.jsonl")}

    print(f"挑 {N_SLOW} 慢 + {N_FAST} 快，重建候选池并对比 max_tokens 2048 vs 256\n", flush=True)
    pool = {}
    for qid, _, _ in picked:
        _, chunks = await retrieve(qmap[qid], DEPTS, LEVEL, kb=KB_STRESS, top_k=5)
        pool[qid] = chunks

    sys_tpl, human_tpl = load_ambiguity_check_templates()
    print(f"{'id':<7}{'组':<4}{'原耗时':<9}{'2048耗时':<10}{'2048长度':<10}{'256耗时':<9}{'256长度':<8}")
    print("-" * 60)
    rows = []
    for qid, orig, tag in picked:
        msgs = [
            {"role": "system", "content": sys_tpl},
            {"role": "user", "content": human_tpl.format(
                question=qmap[qid], candidates=_format_candidates(pool[qid], SNIPPET))},
        ]
        ms_a, len_a = await _call(msgs, 2048, s)
        ms_b, len_b = await _call(msgs, 256, s)
        rows.append({"id": qid, "tag": tag, "orig": orig, "a_ms": ms_a, "a_len": len_a,
                     "b_ms": ms_b, "b_len": len_b})
        print(f"{qid:<7}{tag:<4}{orig:<9.0f}{ms_a:<10.0f}{len_a:<10}{ms_b:<9.0f}{len_b:<8}", flush=True)

    slow = [r for r in rows if r["tag"] == "慢"]
    fast = [r for r in rows if r["tag"] == "快"]
    print(f"\n慢组 2048: 平均耗时 {sum(r['a_ms'] for r in slow)/len(slow):.0f}ms  "
          f"平均输出 {sum(r['a_len'] for r in slow)/len(slow):.0f} 字")
    print(f"快组 2048: 平均耗时 {sum(r['a_ms'] for r in fast)/len(fast):.0f}ms  "
          f"平均输出 {sum(r['a_len'] for r in fast)/len(fast):.0f} 字")
    print(f"\n限长 256 后：>6s 的次数 {sum(1 for r in rows if r['b_ms'] > 6000)}/{len(rows)}"
          f"（未限长时 {sum(1 for r in rows if r['a_ms'] > 6000)}/{len(rows)}）")

    print("\n判据：慢组输出明显更长 → 长尾=生成太长；限长后长尾消失 → 对策=压小判定调用的 max_tokens")
    (ROOT / "logs" / "slowcall_cause.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
