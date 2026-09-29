"""D8 运行时复验：改名+解绑后，`complete()` 是否仍能正确关掉思考链。

评审要求（`docs/local/交流区/交流区.md` §1.12 第 2 步）：回执须附**一次判定调用的验证结果**，
确认 `reasoning_effort=none` 仍被下发、思考仍是关的。

做法：包一层 spy 抓 `litellm.acompletion` 的**实际入参**与**响应字段**，
比"看耗时"可靠——耗时会被网络抖动干扰，字段不会。

用法：python logs/verify_d8_runtime.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402

from src.config.litellm_client import complete  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

caught: list[dict] = []


async def main() -> None:
    s = get_settings()
    print(f"配置：provider={s.llm_provider}  model={s.llm_model}  "
          f"thinking_enabled={s.llm_thinking_enabled}")
    print(f"      llm_api_key 已配置 = {bool(s.llm_api_key)}"
          f"  base_url = {s.llm_base_url}\n")
    if not s.llm_api_key:
        print("❌ LLM_API_KEY 为空——.env 改名没生效？")
        return

    orig = litellm.acompletion

    async def spy(*args, **kwargs):
        resp = await orig(*args, **kwargs)
        m = resp.choices[0].message
        caught.append({
            "model": kwargs.get("model"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "drop_params": kwargs.get("drop_params"),
            "finish": resp.choices[0].finish_reason,
            "content_len": len(m.content or ""),
            "reasoning_len": len(getattr(m, "reasoning_content", None) or ""),
        })
        return resp

    litellm.acompletion = spy  # type: ignore[assignment]
    try:
        out = await complete([{"role": "user", "content": "只回复两个字：收到"}], temperature=0)
    finally:
        litellm.acompletion = orig  # type: ignore[assignment]

    print(f"返回内容：{out[:60]!r}\n")
    for c in caught:
        print("实际下发的调用参数与响应：")
        for k, v in c.items():
            print(f"    {k:<18} = {v!r}")

    ok = caught and caught[0]["reasoning_effort"] == "none" and caught[0]["reasoning_len"] == 0
    print("\n判据：")
    print("  · `model` 前缀与 LLM_PROVIDER 一致\n"
          "  · `reasoning_effort='none'` 确实被下发\n"
          "  · 响应 `reasoning_content` 为 0 字 → ✅ 思考仍是关的")
    print("\n" + ("✅ 复验通过：解绑后行为不变" if ok else "❌ 复验未通过，见上方字段"))


if __name__ == "__main__":
    asyncio.run(main())
