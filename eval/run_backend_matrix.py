"""§8 后端矩阵实验：一次性钉死「思考链」与「精排」各自对**延迟**与**质量**的贡献。

⚠️ 本脚本**托管 API 生命周期**——不再人肉重启（优化方案 §8.2 / 协议 §九 P-2：
"启动显式传参 + 事后核实实际生效值"，本项目已因手滑漏传产出过一次错误的 A/B）。

按作用层级拆分（O1 修正后的表达）：
  · `LLM_THINKING_ENABLED` / `RERANK_BACKEND` = **进程级** → 每组重启 API
  · `use_rerank` = **请求级** → 请求体传参
  ⚠️ **"不精排"必须用 `use_rerank=false` 表达，不能用 `RERANK_BACKEND=off`**
     —— `off` 不是合法值，会静默落到 `LLMReranker`（6 组矩阵会从第一格就错）。

矩阵（§8.1）：思考链 {false,true} × 精排 {off, local, llm} = 6 组。

指标（§8.3）：
  · 延迟：**p50 / p90 / p95 / max + >5s 占比**（不许只报平均——曾因"平均 4120ms"掩盖双峰）
  · 质量：R@1 / R@3 / R@5 / MRR（用清晰题 80 条）
  · **完整性：`rerank_empty` / `ambiguity_unparsed` 计数** ← 用来判定"是不是又在空转"
  · 判定类（漏报/误报）：**留空**，阻塞于 D1（地面真值未重标，是错指标）

用法：
    python eval/run_backend_matrix.py --n 8        # 每组 8 题（先小样验证脚本）
    python eval/run_backend_matrix.py              # 全量（清晰题 80）
    python eval/run_backend_matrix.py --groups 0,1 # 只跑指定组
    python eval/run_backend_matrix.py --no-restart # 假定 API 已按当前 env 起好（调试用）

产出：汇总表刷终端；逐题明细写 `logs/matrix_detail.json`（不进上下文）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:8000"
PYTHON = sys.executable
LOG_DIR = ROOT / "logs"
DETAIL_FILE = LOG_DIR / "matrix_detail.json"

# (组名, API进程env, 请求级参数)
GROUPS: list[tuple[str, dict, dict]] = [
    ("思考关 + 精排off",   {"LLM_THINKING_ENABLED": "false", "RERANK_BACKEND": "local"}, {"use_rerank": False}),
    ("思考关 + local",     {"LLM_THINKING_ENABLED": "false", "RERANK_BACKEND": "local"}, {"use_rerank": True}),
    ("思考关 + llm",       {"LLM_THINKING_ENABLED": "false", "RERANK_BACKEND": "llm"},   {"use_rerank": True}),
    ("思考开 + 精排off",   {"LLM_THINKING_ENABLED": "true",  "RERANK_BACKEND": "local"}, {"use_rerank": False}),
    ("思考开 + local",     {"LLM_THINKING_ENABLED": "true",  "RERANK_BACKEND": "local"}, {"use_rerank": True}),
    ("思考开 + llm",       {"LLM_THINKING_ENABLED": "true",  "RERANK_BACKEND": "llm"},   {"use_rerank": True}),
]


def _load(name: str) -> list[dict]:
    p = EVAL_DIR / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def _kill_api() -> None:
    """按端口找 PID 并杀掉（Windows）。"""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return
    for line in out.splitlines():
        if "LISTENING" in line and ":8000" in line:
            pid = line.split()[-1]
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=30)
            break
    time.sleep(2)


def _start_api(env_extra: dict, log_path: Path) -> subprocess.Popen:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",          # 跳过 HF 联网检查（本地模型加载快很多）
        **env_extra,
    }
    fh = open(log_path, "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
    )


def _wait_health(timeout: int = 90) -> bool:
    for _ in range(timeout // 3):
        try:
            if httpx.get(f"{BASE}/health", timeout=3).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def _ask(question: str, token: str, req_kw: dict, kb: str, with_answer: bool = True) -> dict | None:
    """打一次 /demo/ask。

    ⚠️ **必须显式传 kb**：`/demo/ask` 的 kb 默认是**真实库**（空的），
    而评测题集在**压测库** —— 不传就会问空库、全部 0 命中。
    （小样试跑就是这么抓到 R@1=0.000 的；顺带发现"空库→0 候选→精排根本没被调用"。）
    """
    try:
        r = httpx.post(f"{BASE}/demo/ask", json={
            "question": question, "token": token, "kb": kb, "with_answer": with_answer, **req_kw,
        }, timeout=300)
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def _count_audit(log_path: Path) -> dict:
    """从该组的 API 日志里数空输出事件（§8.3 的"完整性"列）。"""
    out = {"rerank_empty": 0, "ambiguity_unparsed": 0, "rerank": 0, "ambiguity_check": 0}
    if not log_path.exists():
        return out
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    for k in out:
        out[k] = len(re.findall(rf'"event":\s*"{k}"', txt))
    return out


def run_group(name: str, env_extra: dict, req_kw: dict, rows: list[dict], token: str,
              kb: str, warmup: bool = True) -> dict:
    log_path = LOG_DIR / f"matrix_{name.replace(' ', '_').replace('+', '')}.log"
    print(f"\n{'=' * 72}\n组：{name}\n  env={env_extra}  请求级={req_kw}  kb={kb}", flush=True)

    _kill_api()
    _start_api(env_extra, log_path)
    if not _wait_health():
        print("  ❌ API 未就绪，跳过该组")
        return {"group": name, "error": "api_not_ready"}
    print(f"  API 就绪（日志 {log_path.name}）", flush=True)

    # 预热（协议 §九 P-3：冷态 3018ms vs 热态 538ms，差 5.6 倍；本地精排首次还要载模型）
    if warmup:
        # 连打 2 次：第 1 次可能还在加载本地模型/冷启动，第 2 次才是真正的热态。
        # （小样试跑时 local 组预热 25s，首次请求仍偏高——单次预热不够。）
        t0 = time.perf_counter()
        d0 = _ask("预热", token, req_kw, kb)
        t1 = time.perf_counter()
        _ask("预热第二次", token, req_kw, kb)
        t2 = time.perf_counter()
        # P-2「事后核实实际生效值」：确认 kb 与精排后端确实是这一组的配置
        eff = (d0 or {}).get("kb")
        backend = ((d0 or {}).get("rerank") or {}).get("backend")
        print(f"  预热完成（首次 {t1 - t0:.0f}s / 第二次 {t2 - t1:.1f}s）"
              f"  ｜ 生效 kb={eff} 精排后端={backend}", flush=True)

    lat_total, lat_ret, lat_amb = [], [], []
    ranks: list[int | None] = []
    detail = []
    for i, r in enumerate(rows, 1):
        d = _ask(r["question"], token, req_kw, kb)
        if d is None:
            detail.append({"q": r["question"], "error": True})
            continue
        t = d.get("timings", {})
        lat_total.append(t.get("total_ms", 0))
        lat_ret.append(t.get("retrieve_ms", 0))
        lat_amb.append(t.get("ambiguity_ms", 0))
        # 质量：期望来源在最终结果里的名次（1-based）
        rank = next((j for j, x in enumerate(d.get("final", []), 1) if x["source"] == r["source"]), None)
        ranks.append(rank)
        detail.append({
            "q": r["question"], "expect": r["source"], "rank": rank,
            "kb": d.get("kb"), "timings": t,
            "final_sources": [x["source"] for x in d.get("final", [])],
        })
        if i % 10 == 0:
            print(f"    {i}/{len(rows)} ...", flush=True)

    n = len(ranks)
    res = {
        "group": name, "env": env_extra, "req": req_kw, "n": n,
        "lat_total": {"p50": _pct(lat_total, .5), "p90": _pct(lat_total, .9),
                      "p95": _pct(lat_total, .95), "max": max(lat_total) if lat_total else 0,
                      "gt5s": sum(1 for x in lat_total if x > 5000) / len(lat_total) if lat_total else 0},
        "lat_ret_p50": _pct(lat_ret, .5),
        "lat_amb_p50": _pct(lat_amb, .5),
        "mrr": sum(1.0 / r for r in ranks if r) / n if n else 0.0,
        **{f"R@{k}": sum(1 for r in ranks if r and r <= k) / n if n else 0.0 for k in (1, 3, 5)},
        "audit": _count_audit(log_path),
    }
    print(f"  → p50={res['lat_total']['p50']}ms p90={res['lat_total']['p90']}ms "
          f"R@1={res['R@1']:.3f} MRR={res['mrr']:.3f} "
          f"空输出(rerank/amb)={res['audit']['rerank_empty']}/{res['audit']['ambiguity_unparsed']}",
          flush=True)
    res["_detail"] = detail
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="每组跑多少题（0=全量清晰题）")
    ap.add_argument("--groups", default="", help="只跑指定组序号，如 0,1")
    ap.add_argument("--token", default="demo-it-token")
    ap.add_argument("--kb", default="stress",
                    help="目标知识库。默认 stress —— 评测题集跑在**合成语料**上，而它属于压测库；"
                         "真实库是空的，问它会全部 0 命中")
    ap.add_argument("--no-restart", action="store_true", help="不重启 API（假定已按当前 env 起好）")
    args = ap.parse_args()

    global _kill_api, _start_api, _wait_health  # noqa: PLW0603
    if args.no_restart:
        _kill_api = lambda: None          # noqa: E731
        _start_api = lambda *a, **k: None  # noqa: E731
        _wait_health = lambda *a, **k: True  # noqa: E731

    rows = _load("dataset_clear.jsonl")
    if args.n:
        rows = rows[: args.n]
    if not rows:
        print("❌ 找不到 dataset_clear.jsonl")
        return

    idx = [int(x) for x in args.groups.split(",") if x.strip()] if args.groups else list(range(len(GROUPS)))
    print(f"矩阵实验：{len(idx)} 组 × {len(rows)} 题（清晰题）")
    print("⚠️ 判定类指标（漏报/误报）**留空** —— 阻塞于 D1（地面真值未重标）")

    results = []
    for i in idx:
        name, env_extra, req_kw = GROUPS[i]
        results.append(run_group(name, env_extra, req_kw, rows, args.token, args.kb))

    # ---- 汇总表（§8.6：只把汇总刷终端，明细写文件）----
    print(f"\n{'=' * 96}")
    print(f"{'组':<16}{'p50':<8}{'p90':<8}{'p95':<8}{'max':<8}{'>5s':<8}"
          f"{'R@1':<8}{'R@3':<8}{'R@5':<8}{'MRR':<8}{'空输出':<8}")
    print("-" * 96)
    for r in results:
        if r.get("error"):
            print(f"{r['group']:<16}❌ {r['error']}")
            continue
        lt = r["lat_total"]
        print(f"{r['group']:<16}{lt['p50']:<8}{lt['p90']:<8}{lt['p95']:<8}{lt['max']:<8}"
              f"{lt['gt5s']:<8.0%}{r['R@1']:<8.3f}{r['R@3']:<8.3f}{r['R@5']:<8.3f}"
              f"{r['mrr']:<8.3f}{r['audit']['rerank_empty']}/{r['audit']['ambiguity_unparsed']}")

    detail = {r["group"]: r.pop("_detail", []) for r in results if not r.get("error")}
    LOG_DIR.mkdir(exist_ok=True)
    DETAIL_FILE.write_text(json.dumps({"summary": results, "detail": detail},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n逐题明细已写入 {DETAIL_FILE}（不进上下文，需要时再挑着看）")
    print("\nR1 判据（§8.4 按 O3 修正）：看「**思考开 + llm**」组的 `rerank_empty` 计数——")
    print("  >0 → 历史 88.8% 确实产自空转，R1 成立、标注作废；=0 → R1 暂缓。")


if __name__ == "__main__":
    main()
