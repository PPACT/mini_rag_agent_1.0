"""`2.0-50` 决定性那一步：**灌库结果可复现吗？** —— 在**入库层**量，不在解析层猜。

## 为什么要在这一层量

文档侧的担心是「**同一份代码、同一批语料，跑两次得到不同的库**」——
那说的**不是"某个文件会不会崩"，是"库一不一样"**。

⚠️ 所以在解析层数"崩了几次"**答不了这个问题**：
解析成功≠入库一致（切块、向量、写库都可能引入差异）。

## ⭐ 它靠 `2.0-30` 的 `content_hash` 才做得出来

跑一次 ingest，就记一份库的**指纹**：

```
内容指纹 = md5(按 (文件, 块序号) 排序后的 content_hash 拼接)
向量指纹 = md5(按 (文件, 块序号) 排序后的 embedding 维度与范数四舍五入)
清单     = {文件: 块数}
```

**每一遍都记，最后比对 n 遍的指纹是否完全一致。**

⚠️ 用 `content_hash`（而不是 `content` 原文）作指纹，是**刻意的**：
它**短、可比对**，而且**它正是"要不要重嵌"的判据** ——
指纹一致 ⇔ 没有任何块需要重嵌 ⇔ **构建可复现**。

用法：
    python eval/check_build_reproducible.py --runs 5 --kb stress --dir docs/corpus
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.connection import close_pool, get_pool  # noqa: E402


async def _fingerprint(kb: str) -> dict:
    """库的指纹：内容 / 向量 / 清单。

    ⚠️ 排序键固定为 `(source_file, chunk_index)` —— **不排的话，行序一变指纹就变**，
    那是假的"不可复现"。
    """
    pool = await get_pool(kb)
    rows = await pool.fetch(
        """
        SELECT d.original_name AS f, c.chunk_index AS i, c.content_hash AS h,
               vector_dims(c.embedding) AS dim,
               round(sqrt(inner_product(c.embedding, c.embedding))::numeric, 4) AS norm
          FROM chunks c JOIN documents d ON d.id = c.document_id
         ORDER BY d.original_name, c.chunk_index
        """
    )
    content = hashlib.md5("|".join(r["h"] or "" for r in rows).encode()).hexdigest()
    vec = hashlib.md5("|".join(f"{r['dim']}:{r['norm']}" for r in rows).encode()).hexdigest()
    manifest = {}
    for r in rows:
        manifest[r["f"]] = manifest.get(r["f"], 0) + 1
    return {"n_chunks": len(rows), "content_md5": content, "vector_md5": vec,
            "manifest_md5": hashlib.md5(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "n_files": len(manifest), "files_without_hash": sum(1 for r in rows if not r["h"])}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--kb", default="stress")
    ap.add_argument("--dir", default=str(ROOT / "docs" / "corpus"))
    ap.add_argument("--json", default=str(ROOT / "logs" / "build_reproducible.json"))
    ap.add_argument("--until-failure", action="store_true",
                    help="一直跑到**抓住一次失败**（或到 --max-runs）—— 失败率低时不必靠运气")
    ap.add_argument("--max-runs", type=int, default=30)
    args = ap.parse_args()

    from eval.ingest import ingest, truncate   # 复用**同一条**入库路径（不另写一份）
    from src.document_parser.routing import failure_report

    runs = args.max_runs if args.until_failure else args.runs
    print(f"2.0-50 ｜ 灌库可复现性 ｜ {args.kb} ← {args.dir} ｜ 最多 {runs} 遍"
          f"{'（抓到一次失败就停）' if args.until_failure else ''}")
    print("⚠️ 每遍都 `--truncate` 重灌；指纹用 `2.0-30` 的 `content_hash` +\n"
          "   向量的「维度:范数」（向量不逐位比对：浮点末位抖动不算构建不可复现）\n")

    prints: list[dict] = []
    seen_fail_reasons: list[str] = []
    for r in range(1, runs + 1):
        await truncate(args.kb)
        # ⭐ 把失败清单接出来 —— `2.0-50` 要的是"**哪一篇、什么原因**"，光有"少了几块"没用
        fails: list = []
        await ingest(False, [Path(args.dir)], args.kb, failures_out=fails)
        fp = await _fingerprint(args.kb)
        prints.append(fp)
        tag = "" if fp["n_chunks"] == max((p["n_chunks"] for p in prints), default=0) else "  ← 少了块"
        print(f"  第 {r} 遍: 块 {fp['n_chunks']} ｜ 文件 {fp['n_files']} ｜ "
              f"内容 {fp['content_md5'][:12]} ｜ 向量 {fp['vector_md5'][:12]}{tag}")
        for e in fails:
            reason = f"{Path(e.path).name} ｜ {e.reason}"
            if reason not in seen_fail_reasons:
                seen_fail_reasons.append(reason)
                print(f"    ❌ {reason}")
                print("      " + failure_report(e).replace("\n", "\n      "))
        if args.until_failure and r > 1 and len({p["n_chunks"] for p in prints}) > 1:
            print(f"\n  → 已抓住不一致（第 {r} 遍），停")
            break

    print("\n【结论】")
    for key, label in (("n_chunks", "块数"), ("content_md5", "内容指纹"),
                       ("vector_md5", "向量指纹"), ("manifest_md5", "清单指纹")):
        vals = {p[key] for p in prints}
        print(f"  {label}: {'✅ 每遍一致' if len(vals) == 1 else f'❌ 出现 {len(vals)} 种不同值！'}")
    same = len({(p["content_md5"], p["vector_md5"], p["manifest_md5"]) for p in prints}) == 1
    print(f"\n  → {'✅ 构建**可复现**（同代码同语料，每遍得到同一个库）' if same else '❌ 构建**不可复现** —— 必须先定位再谈 2.0-34'}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kb": args.kb, "dir": args.dir, "runs": len(prints),
                               "fingerprints": prints, "reproducible": same,
                               "fail_reasons": seen_fail_reasons},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    # ⚠️ `--json` 可能是**相对路径** → `relative_to` 会抛（实测崩过一次，白等 8 分钟）
    try:
        shown: object = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\n产物 → {shown}")
    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
