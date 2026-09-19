"""测量歧义判定 LLM 调用的**长尾**，并判断长尾是不是并发造成的。

背景（来自 `measure_ambiguity_snippet.py` 的实测）：
  - 平均值（4.6s）远高于中位数（3.2s）→ **尾部在拖平均**
  - 每一档都有 8~11 次调用 >6s（占 ~22%），且**各档慢的不是同一批题**（交集仅 1）
  → 长尾与 prompt 长度无关（snippet 缩短无效）。

但还有一个未排除的可能：**长尾是我自己开并发(5)造成的**（限流 → 排队/重试）。
若不排除就下结论，可能是自伤。

本脚本用**固定 prompt**（不检索、不涉及 snippet），只对比 串行 vs 并发5：
  - 两者长尾都明显 → 长尾是 **API 侧固有** → 可用「超时上限 + 失败回退」兜住
  - 只有并发时长尾   → 长尾是 **自伤的** → 线上单发调用不受影响，别乱改

用法：python eval/measure_llm_tail.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.litellm_client import complete  # noqa: E402

SERIAL_N = 20        # 串行次数
CONC_N = 20          # 并发次数
CONCURRENCY = 5      # 并发度（与测量脚本一致）
SLOW_MS = 6000       # 长尾阈值

# 固定 prompt：模拟真实的歧义判定输入（5 条 × 300 字候选）
_CAND = "".join(f"[{i}] (来源: 员工手册v{i}.pdf)\n" + "本制度自发布之日起施行，具体解释权归人力资源部。" * 17 + "\n\n"
                for i in range(1, 6))
PROMPT = [{"role": "user", "content": "判断以下候选之间是否存在互相矛盾的答案，输出 JSON。\n\n" + _CAND}]


def _emit(name: str, lat: list[float]) -> None:
    s = sorted(lat)
    n_slow = sum(1 for x in s if x > SLOW_MS)
    print(f"  {name:<10} n={len(s):<3} p50 {s[len(s)//2]:>6.0f}ms  p90 {s[int(len(s)*0.9)]:>6.0f}ms  "
          f"max {s[-1]:>6.0f}ms  平均 {sum(s)/len(s):>6.0f}ms  >{SLOW_MS//1000}s: {n_slow}/{len(s)}"
          f" ({n_slow/len(s):.0%})", flush=True)


async def _one(sem: asyncio.Semaphore | None) -> float:
    async def _call() -> float:
        t0 = time.perf_counter()
        await complete(PROMPT, temperature=0)
        return (time.perf_counter() - t0) * 1000
    if sem is None:
        return await _call()
    async with sem:
        return await _call()


async def main() -> None:
    print(f"固定 prompt（~{len(PROMPT[0]['content'])} 字），串行 {SERIAL_N} 次 → 并发 {CONCURRENCY} × {CONC_N} 次\n",
          flush=True)

    print("① 串行（线上歧义判定就是单发调用）...", flush=True)
    serial = [await _one(None) for _ in range(SERIAL_N)]
    _emit("串行", serial)

    print(f"\n② 并发 {CONCURRENCY} ...", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    conc = list(await asyncio.gather(*[_one(sem) for _ in range(CONC_N)]))
    _emit(f"并发{CONCURRENCY}", conc)

    # 判据
    sr = sum(1 for x in serial if x > SLOW_MS) / len(serial)
    cr = sum(1 for x in conc if x > SLOW_MS) / len(conc)
    print(f"\n长尾率：串行 {sr:.0%} ｜ 并发 {cr:.0%}（差 {abs(cr-sr):.0%}）")
    if sr == 0 and cr == 0:
        print("→ 两种模式都没出现长尾：**本次未复现**，不能据此说「API 侧固有」。")
        print("  真实长尾的根因已查明 = 推理模型思考吃满 max_tokens（见 measure_slowcall_cause.py）")
    elif abs(cr - sr) < 0.15:
        print("→ 两者接近：**长尾是 API 侧固有**，与并发无关。")
        print("  可行对策：给歧义判定设「超时上限 + 失败回退为不歧义」（97.5% 的调用本就不歧义，回退安全）")
    elif cr > sr:
        print("→ 并发明显更差：**长尾是自伤的**。线上单发调用不受此影响，测量需改小并发重跑。")
    else:
        print("→ 串行反而更差，噪声过大，样本需加大。")


if __name__ == "__main__":
    asyncio.run(main())
