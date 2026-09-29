"""探针：歧义判定的 LLM 响应到底长什么样？（查"空输出"与"10 秒长尾"的根因）

已达成的结论（勿重复验证）：
  1. `deepseek-v4-flash` 是**推理模型**：先出 `reasoning_content`（思考），再出 `content`（JSON 结论），
     **两者共用 `max_tokens` 预算**。
  2. 思考一旦吃满预算 → `finish_reason=length`、`content` 为空 →
     `parse_result("")` 返回"不歧义" → **静默漏报**（2003、2001 在 256/512 档都丢了结论）。
  3. 10 秒长尾 = 生成满 2048 token 的耗时（≈200 token/s）。
  ⇒ **压小 max_tokens 是靠砍掉思考换速度，静默牺牲正确性，方向是反的。**

本脚本验证**正确的**修法方向（这道题用 2048 预算不够）：
  - 提高 max_tokens（4096 / 8192）→ 结论能否产出？代价多少秒？
  - `reasoning_effort="low"` → 能否既产出结论又保持快？（若 provider 不支持会报错，也一并记录）

用法：python eval/probe_llm_response.py
"""
from __future__ import annotations

import asyncio
import json
import sys
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

TARGETS = [("2003", "2048 预算下丢结论"), ("2001", "2048 勉强够")]
DEPTS = ["IT", "公司"]
VARIANTS = [
    ("2048 现状(思考开)", 2048, {}),
    ("2048+effort_none", 2048, {"reasoning_effort": "none"}),
    ("512+effort_none", 512, {"reasoning_effort": "none"}),
    ("2048+thinking_off", 2048, {"thinking": {"type": "disabled"}}),
]


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _probe(msgs: list[dict], max_tokens: int, extra: dict, s) -> None:
    import time
    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=f"{s.llm_provider}/{s.llm_model}", messages=msgs, api_key=s.llm_api_key,
            api_base=s.llm_base_url, temperature=0, max_tokens=max_tokens, timeout=120,
            **extra,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    {extra or 'base':<18} ❌ 调用失败: {str(e)[:110]}", flush=True)
        return
    ms = (time.perf_counter() - t0) * 1000
    c = resp.choices[0]
    msg = c.message
    content = msg.content or ""
    reason = getattr(msg, "reasoning_content", None) or ""
    u = getattr(resp, "usage", None)
    ok = "✅" if content else "❌空"
    print(f"    max={max_tokens:<5}{str(extra or ''):<18} {ms:>6.0f}ms  finish={c.finish_reason:<7} "
          f"completion={getattr(u, 'completion_tokens', '?')!s:<5} content={len(content):<4}字 "
          f"reasoning={len(reason):<5}字 {ok}", flush=True)
    if content:
        print(f"      → {content[:90].replace(chr(10), ' ')}")


async def main() -> None:
    s = get_settings()
    print(f"模型 = {s.llm_model}\n")
    qmap = {str(r["id"]): r["question"]
            for r in _load("dataset_clear.jsonl") + _load("dataset_clarify.jsonl")}
    sys_tpl, human_tpl = load_ambiguity_check_templates()

    for qid, tag in TARGETS:
        _, chunks = await retrieve(qmap[qid], DEPTS, 3, kb=KB_STRESS, top_k=5)
        msgs = [
            {"role": "system", "content": sys_tpl},
            {"role": "user", "content": human_tpl.format(
                question=qmap[qid], candidates=_format_candidates(chunks, 300))},
        ]
        print(f"  {qid} [{tag}] {qmap[qid][:30]}")
        for label, mt, extra in VARIANTS:
            print(f"    {label}", end="", flush=True)
            print("\r", end="")
            await _probe(msgs, mt, extra, s)
        print(flush=True)

    print("判据：")
    print("  · 提高 max_tokens 后 content 出现 → 修法=留够预算（但要付时间代价，看上面 ms）")
    print("  · reasoning_effort=low 若既产出又更快 → 最佳解")
    print("  · 都不行 → 判定环节应换**非推理**模型（分类任务不需要思考链）")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
