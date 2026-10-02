"""`2.0-50` 的**可见化**那一步：**评估前核对"灌库完整性"**。

## 为什么是"可见"而不是"修"

`03_*.pdf` 有 **≈2%** 的概率在 `ingest` 流程里丢掉（`§1.59` 量的：单文件解析复现不出来，
**只有完整 ingest 才偶发**）。**诱因未知 → 修它不是一两天的事。**

→ 所以先做**便宜且立刻有用**的那一步：**让它可见**。

> ⭐ 本项目的取向：**判不准不要紧，别让它悄悄错。**

**判据**：**语料目录里能解析的文件，是不是都在库里？**
⚠️ 失败清单原先**只在 `ingest` 内部打印** —— 跑完就没了，
**而评估是另一条命令**，它不会告诉你"库里少了谁"。**这就是静默的来源。**

用法（**评估前跑**）：
    python eval/check_ingest_complete.py --kb stress --dir docs/local/corpus
退出码非 0 = **不完整**，别拿它做评估。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.paths import CORPUS_DIR  # noqa: E402

from src.db.connection import close_pool, get_pool  # noqa: E402


def compare(corpus: set[str], in_db: set[str]) -> dict[str, list[str]]:
    """纯函数：语料 vs 库里的差集。

    ⚠️ **两边都列出来**（不只列"缺的"）——
    库里**多出**语料目录之外的文件，同样会让评估数字解释不通。

    ⚠️ 传进来的 `corpus` 必须是**路由认得的**那些 —— 见 `split_by_routing`。
    """
    return {"missing": sorted(corpus - in_db), "unexpected": sorted(in_db - corpus)}


def split_by_routing(files: list[Path]) -> tuple[set[str], set[str]]:
    """把语料分成「**路由认得的**」与「**路由明确拒收的**」。

    ⭐ **这一步不做，闸就是废的**：`docs/local/corpus` 里有一份 `.html`，
    而 `routing` **明确拒收 HTML**（`2.0-40` 未接入）——
    不把它摘出去，这个闸**每次都报"不完整"** → **永远红 = 和永远绿一样没用**。

    ⚠️ 判定**复用 `route_file`**（不是自己再列一份格式清单）——
    那份清单只能有一个来源（`routing._KNOWN_UNSUPPORTED`）。
    """
    from src.document_parser.routing import DocumentParseError, route_file

    known: set[str] = set()
    unsupported: set[str] = set()
    for p in files:
        try:
            route_file(str(p))
            known.add(p.name)
        except DocumentParseError:
            unsupported.add(p.name)   # 路由**明确**说"我不认这个格式" → 预期跳过
    return known, unsupported


async def check(kb: str, directory: str) -> dict:
    """闸的核心 —— **抽出来是给别的脚本复用的**（评估前必须过）。

    ⚠️ 为什么不只是"文档里写一句'评估前记得跑"：
    **记在文档里的规矩拦不住遗忘**（本项目已踩过：`优化方案.md` 早写了 `use_hybrid` 的坑，
    照样又踩一次）→ **让评估脚本自己调它**，才真的忘不掉。
    """
    from eval.ingest import collect_files

    corpus_dir = Path(directory)
    files = collect_files(False, [corpus_dir])
    corpus, unsupported = split_by_routing(files)

    pool = await get_pool(kb)
    try:
        rows = await pool.fetch(
            "SELECT COALESCE(original_name, filename) AS n, chunk_count FROM documents "
            "WHERE is_deleted = false")
    finally:
        await close_pool()

    in_db = {r["n"] for r in rows}
    return {
        "files": len(files), "known": len(corpus), "unsupported": sorted(unsupported),
        "in_db": len(in_db), "empty": sorted(r["n"] for r in rows if not r["chunk_count"]),
        **compare(corpus, in_db),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--dir", default=str(CORPUS_DIR))
    args = ap.parse_args()

    from src.db.kb import validate
    validate(args.kb)

    r = await check(args.kb, args.dir)
    print(f"灌库完整性 ｜ kb={args.kb} ｜ 语料 {r['files']} 个文件"
          f"（路由认得 {r['known']} / 明确拒收 {len(r['unsupported'])}）｜ 库中 {r['in_db']} 份文档")
    if r["unsupported"]:
        # ⚠️ **不是"丢了"，是"路由不认这个格式"** —— 分开报，免得跟真丢的混在一起
        print(f"  ⏭️ 路由明确拒收（预期跳过，不算失败）：{', '.join(r['unsupported'])}")
    for key, bad, good, what in (
        ("missing", "❌ 丢了（语料里有、库里没有）", "✅ 没丢", "语料里的文件都在库里"),
        ("unexpected", "⚠️ 多出来（库里有、语料目录里没有）", "✅ 没多", "库里没有语料之外的文件"),
    ):
        if r[key]:
            print(f"\n{bad}：{len(r[key])}")
            for n in r[key]:
                print(f"    {n}")
        else:
            print(f"  {good}：{what}")

    if r["empty"]:
        print(f"\n⚠️ 入库但**块数为 0** 的：{len(r['empty'])}")
        for n in r["empty"]:
            print(f"    {n}")

    ok = not r["missing"] and not r["unexpected"] and not r["empty"]
    print("\n" + ("✅ 完整 —— 可以拿这个库做评估"
                  if ok else "❌ **不完整** —— ⚠️ 先别用它做评估，数字会解释不通"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
