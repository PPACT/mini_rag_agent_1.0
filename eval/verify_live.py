"""端到端真实链路验证：通过 HTTP 调 /demo/ask，看**真实体验**（通过率 + 单次延迟）。

与其它评测的区别：这是**串行**跑完整链路（含 LLM 生成），
所以它测的是"用户真实等多久"，而非并发平均。

前置：服务已起（uvicorn src.main:app），语料已灌入。

用法：
    python eval/verify_live.py                 # 默认 15 题
    python eval/verify_live.py --n 30          # 跑更多
    python eval/verify_live.py --no-answer     # 只测检索，不生成（快）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="跑多少题")
    ap.add_argument("--token", default="demo-it-token")
    ap.add_argument("--no-answer", action="store_true", help="不生成答案（只测检索，快）")
    ap.add_argument("--dataset", default="dataset_clear.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (EVAL_DIR / args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()][: args.n]

    ok, lat_ret, lat_total = 0, [], []
    fails = []
    print(f"{'#':<4}{'问题':<30}{'检索(ms)':<10}{'总(ms)':<10}{'结果'}")
    print("-" * 72)
    for i, r in enumerate(rows, 1):
        try:
            resp = httpx.post(f"{BASE}/demo/ask", json={
                "question": r["question"], "token": args.token,
                "with_answer": not args.no_answer,
            }, timeout=300)
            d = resp.json()
        except Exception as e:  # noqa: BLE001
            print(f"{i:<4}{r['question'][:28]:<30}{'-':<10}{'-':<10}异常 {e}")
            continue

        t = d["timings"]
        lat_ret.append(t["retrieve_ms"])
        lat_total.append(t["total_ms"])
        # 判定：最终结果里是否命中期望来源
        hit = any(x["source"] == r["source"] for x in d["final"])
        if hit:
            ok += 1
        else:
            fails.append((r["question"], r["source"], [x["source"][:18] for x in d["final"]][:3]))
        print(f"{i:<4}{r['question'][:28]:<30}{t['retrieve_ms']:<10}{t['total_ms']:<10}{'✅' if hit else '❌'}")

    n = len(rows)
    print("-" * 72)
    print(f"来源命中率: {ok}/{n} = {ok / n:.1%}" if n else "无数据")
    if lat_ret:
        print(f"检索延迟: 平均 {sum(lat_ret) / len(lat_ret):.0f}ms  最大 {max(lat_ret)}ms")
    if lat_total and not args.no_answer:
        print(f"端到端延迟: 平均 {sum(lat_total) / len(lat_total):.0f}ms  最大 {max(lat_total)}ms")
    if fails:
        print("\n未命中明细（期望来源 vs 实际 top-3）:")
        for q, exp, got in fails:
            print(f"  {q[:26]:<28} 期望 {exp[:20]} | 实际 {got}")


if __name__ == "__main__":
    main()
