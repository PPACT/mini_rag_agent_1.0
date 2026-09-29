"""验证：换成 `openai/` 适配器后，`reasoning_effort="none"`（关思考）是否仍生效。

背景（`docs/local/交流区/交流区.md` §1.12 风险①）：
判定与精排靠 `reasoning_effort="none"` 关掉思考链。**该参数是 DeepSeek 语义**。
若把 litellm 的 provider 前缀从 `deepseek/` 换成通用的 `openai/`，
litellm 可能因 `drop_params=True` **静默丢弃它** → 思考链悄悄复活
→ 判定空输出（漏报）、精排空转（正是 `7d16cc3` 修掉的那个 bug）。

**判据（看原始响应字段，不看耗时）**：
  · `reasoning_content` 为空 且 `content` 非空 → ✅ 思考确实关着
  · `reasoning_content` 非空                  → ❌ 思考复活了
  · `content` 为空                            → ❌ 更糟：结论根本没产出

四组对照（同一 prompt、`temperature=0`、`drop_params=True`，与生产一致）：
  ① `deepseek/` + effort=none   ← 生产现状（基准，预期"关"）
  ② `openai/`   + effort=none   ← **待验证：问题就在这一格**
  ③ `openai/`   + 不传 effort   ← 对照（预期"开"）
  ④ `deepseek/` + 不传 effort   ← 对照（预期"开"）

用法：python logs/verify_provider_unbind.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import litellm  # noqa: E402

from src.config.prompts import load_ambiguity_check_templates  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

# 造一个**真会引发矛盾判断**的候选集（两来源数值互斥）——思考链若是开的，会明显长篇推理
CANDIDATES = """[1] (来源: security_policy.md)
员工账号密码每 90 天强制更换一次，逾期未改将锁定账号。

[2] (来源: dept00_security_policy.md)
员工账号密码每 91 天强制更换一次，逾期未改将锁定账号。

[3] (来源: llm047_监控告警.md)
P1（核心业务中断）须 5 分钟内响应、15 分钟内启动处置、2 小时内恢复。

[4] (来源: dept00_customer_service.md)
P1 级（系统不可用）16 分钟内响应，3 小时内给出方案。

[5] (来源: 员工手册.md)
年假按工龄计算：满 1 年 5 天，满 10 年 10 天。
"""

VARIANTS = [
    ("① deepseek/ + effort=none（生产现状）", "deepseek/{m}", {"reasoning_effort": "none"}),
    ("② openai/   + effort=none  ← 待验证",   "openai/{m}",   {"reasoning_effort": "none"}),
    ("③ openai/   + 不传 effort",             "openai/{m}",   {}),
    ("④ deepseek/ + 不传 effort",             "deepseek/{m}", {}),
]


async def call(model: str, extra: dict, msgs: list[dict], s) -> None:
    t0 = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=msgs,
            api_key=s.deepseek_api_key,
            api_base=s.deepseek_base_url,
            temperature=0,
            max_tokens=2048,
            timeout=120,
            drop_params=True,      # 与生产一致：未支持的参数会被静默丢弃
            **extra,
        )
    except Exception as e:  # noqa: BLE001
        print(f"   ❌ 调用失败: {str(e)[:130]}", flush=True)
        return
    ms = (time.perf_counter() - t0) * 1000
    c = resp.choices[0]
    content = c.message.content or ""
    reason = getattr(c.message, "reasoning_content", None) or ""
    if not content:
        mark = "❌ 结论未产出"
    elif reason:
        mark = "❌ 思考复活"
    else:
        mark = "✅ 思考关闭"
    print(f"   {ms:>6.0f}ms  finish={c.finish_reason:<7} "
          f"content={len(content):<4}字 reasoning={len(reason):<5}字  {mark}", flush=True)
    if content:
        print(f"      → {content[:80].replace(chr(10), ' ')}", flush=True)
    if reason:
        print(f"      reasoning前80字: {reason[:80].replace(chr(10), ' ')}", flush=True)


async def main() -> None:
    s = get_settings()
    sys_tpl, human_tpl = load_ambiguity_check_templates()
    msgs = [
        {"role": "system", "content": sys_tpl},
        {"role": "user", "content": human_tpl.format(
            question="密码多久要换一次？候选里 P1 响应时限是多少？", candidates=CANDIDATES)},
    ]
    print(f"底层模型名 = {s.deepseek_model}｜api_base = {s.deepseek_base_url}\n")
    for label, model_tpl, extra in VARIANTS:
        print(label)
        await call(model_tpl.format(m=s.deepseek_model), extra, msgs, s)
        print(flush=True)

    print("判读：")
    print("  · ② 与 ① 一致（思考关闭）→ **可全用 openai/，彻底厂商中立**")
    print("  · ② 出现 reasoning → 换 openai/ 会让思考复活 → **必须保留 deepseek 适配器**（按 D8 原方案）")


if __name__ == "__main__":
    asyncio.run(main())
