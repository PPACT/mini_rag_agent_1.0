"""测量「缩短歧义判定 prompt」的收益与代价。

背景：歧义判定是最大延迟瓶颈（1.8~2.3s / 56.7%），门控方案已被数据排除。
剩下可试的：**缩短送给判定 LLM 的候选文本**（`ambiguity_snippet_chars`，当前 300 字）。
更短的 prompt → LLM 处理更少 token → 更快，但**可能影响判定质量**。

方法（控制变量）：**候选池固定**（只检索一次），仅改变 snippet 长度，
对每档记录：① 平均耗时 ② 漏报率 ③ 误报率 ④ 与 300 字基准的判定一致率。

⚠️ 两处测量卫生（上一版踩的坑，本版已修）：
1. **并发**：上一版全串行，45 题 × 4 档 = 180 次 LLM 调用，跑 15 分钟没完。
   本版用 `asyncio.Semaphore` 并发（判定是网络 IO，并发不改变单次耗时量级）。
2. **剔除「门控题」**：`check_ambiguity` 里尚存一道 `is_diverse` 门控，不通过时
   **根本不调 LLM**（耗时≈0、verdict 恒 False）。留着会①稀释平均耗时
   ②把门控造成的漏报错算到 snippet 头上。本版只保留**真正会调 LLM**的题，
   并打印被剔除的数量（若剔除过多，说明门控本身值得复查）。

用法：python eval/measure_ambiguity_snippet.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.ambiguity import check_ambiguity, is_diverse  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402

LEVELS = (300, 150, 100, 60)     # snippet 字数档位（300 是当前值，作为基准）
N_CLEAR = 20                     # 清晰题取样（应判"不歧义"）
N_AMBIG = 15                     # 歧义题取样（应判"歧义"）
N_ADV = 10                       # 对抗题取样（应判"歧义"）
DEPTS = ["IT", "公司"]
LEVEL = 3
CONCURRENCY = 5                  # 判定并发
POOL_CONCURRENCY = 4             # 候选池构建并发（每题含一次 LLM 精排）
OUT_FILE = Path("logs/snippet_measure.json")   # 逐题明细（gitignored），只把汇总打进上下文


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


async def _build_pool(sem: asyncio.Semaphore, r: dict) -> tuple[str, list]:
    async with sem:
        _, chunks = await retrieve(r["question"], DEPTS, LEVEL, kb=KB_STRESS, top_k=5)
    return r["id"], chunks


async def _judge(sem: asyncio.Semaphore, question: str, chunks: list) -> tuple[float, bool]:
    async with sem:
        t0 = time.perf_counter()
        res = await check_ambiguity(question, chunks)
        return (time.perf_counter() - t0) * 1000, bool(res.ambiguous)


def _pct(n: int, d: int) -> float:
    return n / d if d else 0.0


async def main() -> None:
    s = get_settings()
    clear = _load("dataset_clear.jsonl")[:N_CLEAR]
    clar = _load("dataset_clarify.jsonl")
    ambiguous = [r for r in clar if r.get("kind") == "ambiguous"][:N_AMBIG]
    adversarial = [r for r in clar if r.get("kind") == "adversarial"][:N_ADV]

    # (问题, 期望是否歧义)
    cases = ([(r, False) for r in clear]
             + [(r, True) for r in ambiguous]
             + [(r, True) for r in adversarial])
    print(f"题集：不歧义 {len(clear)} / 歧义 {len(ambiguous)} / 对抗 {len(adversarial)}"
          f" = {len(cases)} 题，{len(LEVELS)} 档 snippet"
          f"（并发 {CONCURRENCY}）\n", flush=True)

    # ① 预先检索候选池（固定，避免重跑检索引入噪声）
    print(f"检索候选池（只做一次，并发 {POOL_CONCURRENCY}；每题含 LLM 精排）...", flush=True)
    t0 = time.perf_counter()
    psem = asyncio.Semaphore(POOL_CONCURRENCY)
    pool = dict(await asyncio.gather(*[_build_pool(psem, r) for r, _ in cases]))
    print(f"  完成，耗时 {time.perf_counter() - t0:.0f}s\n", flush=True)

    # ② 剔除「门控题」——它们不调 LLM，留着会污染耗时与漏报统计
    thr = s.ambiguity_source_threshold
    kept, dropped = [], 0
    for r, should in cases:
        chunks = pool[r["id"]]
        if len(chunks) >= 2 and is_diverse(chunks, thr):
            kept.append((r, should))
        else:
            dropped += 1
    print(f"门控（来源≥{thr}）剔除 {dropped} 题（不调 LLM），保留 {len(kept)} 题"
          f"（不歧义 {sum(1 for _, x in kept if not x)} / 歧义 {sum(1 for _, x in kept if x)}）\n",
          flush=True)
    if not kept:
        print("❌ 无题可测（门控把全部题都拦下了），先检查 ambiguity_source_threshold。")
        await close_pool()
        return

    # ③ 逐档测试
    n_amb = sum(1 for _, x in kept if x)
    n_clr = len(kept) - n_amb
    jsem = asyncio.Semaphore(CONCURRENCY)
    base_verdicts: dict[str, bool] = {}
    rows, detail = [], {}

    for snippet in LEVELS:
        s.ambiguity_snippet_chars = snippet          # 单例，_format_candidates 每次调用时读
        results = await asyncio.gather(
            *[_judge(jsem, r["question"], pool[r["id"]]) for r, _ in kept]
        )
        lat = [x[0] for x in results]
        verdicts = {r["id"]: x[1] for (r, _), x in zip(kept, results)}
        if snippet == LEVELS[0]:
            base_verdicts = verdicts

        miss = sum(1 for r, sh in kept if sh and not verdicts[r["id"]])
        false_pos = sum(1 for r, sh in kept if not sh and verdicts[r["id"]])
        agree = sum(1 for k in verdicts if verdicts[k] == base_verdicts[k]) / len(kept)
        rows.append({
            "snippet": snippet,
            "p50_ms": statistics.median(lat),
            "avg_ms": sum(lat) / len(lat),
            "miss": _pct(miss, n_amb),
            "fp": _pct(false_pos, n_clr),
            "agree": agree,
        })
        detail[str(snippet)] = {r["id"]: {"ms": round(m, 1), "ambiguous": v}
                                for (r, _), (m, v) in zip(kept, results)}
        print(f"  snippet={snippet:>3}: p50 {rows[-1]['p50_ms']:>6.0f}ms  "
              f"平均 {rows[-1]['avg_ms']:>6.0f}ms  漏报 {rows[-1]['miss']:>5.1%}  "
              f"误报 {rows[-1]['fp']:>5.1%}  与基准一致 {agree:>5.1%}", flush=True)

    # ④ 汇总
    base = rows[0]
    print(f"\n{'snippet':<9}{'p50':<9}{'平均':<9}{'漏报率':<9}{'误报率':<9}{'一致率':<9}{'相对300提速'}")
    print("-" * 62)
    for r in rows:
        speed = (1 - r["p50_ms"] / base["p50_ms"]) if base["p50_ms"] else 0
        print(f"{r['snippet']:<9}{r['p50_ms']:<9.0f}{r['avg_ms']:<9.0f}"
              f"{r['miss']:<9.1%}{r['fp']:<9.1%}{r['agree']:<9.1%}{speed:>+7.1%}")

    print("\n解读：找「p50 明显下降 且 漏报/误报不明显变差」的档位。")
    print(f"（说明：300 为当前线上值；一致率=与 300 档判定相同的比例。）")
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps({"summary": rows, "detail": detail}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n逐题明细已写入 {OUT_FILE}（不进上下文，需要时再挑着看）")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
