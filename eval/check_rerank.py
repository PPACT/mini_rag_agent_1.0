"""排查：思考开/关时，LLM 精排到底有没有真的改变顺序？（`eval/` 下，已纳入版本库——**证据脚本不能放会被清理的 `logs/`**）

背景：修复前检索只要 783ms（含一次 LLM 精排调用，不可能这么快）；修复后 1374ms。
若思考开时精排是**抛异常回退**，那"修复前"的链路其实有**两级 LLM 静默降级**。
"""
import asyncio
import sys

sys.path.insert(0, ".")
from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402


async def main() -> None:
    s = get_settings()
    print(f"llm_thinking_enabled = {s.llm_thinking_enabled}", flush=True)
    trace: dict = {}
    _, ch = await retrieve("我迟到了会怎么样？", ["IT", "公司"], 3, top_k=5, trace=trace)
    cands = [f"{c.source_file}#{c.chunk_index}" for c in trace.get("candidates", [])]
    final = [f"{c.source_file}#{c.chunk_index}" for c in trace.get("final", [])]
    print(f"候选 {len(cands)} → 最终 {len(final)}")
    print(f"顺序被改变: {cands[:5] != final}")
    print(f"候选前3: {cands[:3]}")
    print(f"最终   : {final}")
    await close_pool()


asyncio.run(main())
