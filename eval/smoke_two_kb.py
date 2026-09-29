"""冒烟测试：两库分离后，检索是否各查各的（`eval/` 下，已纳入版本库——**证据脚本不能放会被清理的 `logs/`**）。

同时验证：真实库为空时应返回 0 条（**不能因为空就回退去查压测库**）。
"""
import asyncio
import sys

sys.path.insert(0, ".")
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_REAL, KB_STRESS  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

Q = "我迟到了会怎么样？"


async def main() -> None:
    for kb in (KB_STRESS, KB_REAL):
        _, chunks = await retrieve(Q, ["IT", "公司"], 3, kb=kb, use_rerank=False, top_k=5)
        srcs = sorted({c.source_file for c in chunks})
        print(f"kb={kb:<7} 命中 {len(chunks)} 条  来源: {srcs[:3]}")
    await close_pool()


asyncio.run(main())
