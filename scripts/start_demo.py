"""一键启动 RAG 演示（WebUI）。

做四件事：
  1. 确保 Docker 起来，并启动 PG + Redis 容器
  2. 确保 Ollama 在跑（embedding 用）
  3. 启动 FastAPI（若未在跑）
  4. **预热**（发一个请求触发本地精排模型加载，消除首次 30 秒冷启动）
  5. 打开浏览器到 /demo

用法：
    python scripts/start_demo.py            # 用配置里的精排后端
    python scripts/start_demo.py --backend local   # 强制本地 cross-encoder
双击入口：项目根目录的 `启动演示.bat`
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = r"D:\Code_Tools\Miniforge3\envs\mini_pg_agent\python.exe"
DOCKER_DESKTOP = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
OLLAMA_MODELS = "D:/Code_Tools/ollama_models"
OLLAMA_HOST = "127.0.0.1:11451"
API = "http://127.0.0.1:8000"
DEMO_URL = f"{API}/demo"


def log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    # 显式 utf-8 + replace：docker 输出可能含非 GBK 字符，默认编码会抛 UnicodeDecodeError
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        import httpx

        return httpx.get(url, timeout=timeout).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def wait_for(pred, desc: str, timeout: int = 120, interval: float = 3.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(interval)
    log(f"  ⚠ {desc} 超时（{timeout}s），继续尝试下一步")
    return False


def ensure_docker() -> None:
    log("[1/5] 检查 Docker ...")
    if _run(["docker", "ps"]).returncode == 0:
        log("   Docker 已在运行")
    else:
        log("   启动 Docker Desktop（首次可能要 1-2 分钟）...")
        subprocess.Popen([DOCKER_DESKTOP])
        wait_for(lambda: _run(["docker", "ps"]).returncode == 0, "Docker daemon", timeout=180)
        log("   Docker 就绪")

    log("   启动 PG + Redis 容器 ...")
    _run(["docker", "compose", "up", "-d"], cwd=str(PROJECT))
    log("   容器已启动")


def ensure_ollama() -> None:
    log("[2/5] 检查 Ollama（embedding）...")
    if http_ok(f"http://{OLLAMA_HOST}/api/tags"):
        log("   Ollama 已在运行")
        return
    log("   启动 Ollama ...")
    env = {**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS, "OLLAMA_HOST": OLLAMA_HOST}
    subprocess.Popen(
        ["ollama", "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    wait_for(lambda: http_ok(f"http://{OLLAMA_HOST}/api/tags"), "Ollama", timeout=60)
    log("   Ollama 就绪")


def ensure_api(backend: str) -> None:
    log("[3/5] 检查 FastAPI ...")
    if http_ok(f"{API}/health"):
        log("   API 已在运行（若要切换精排后端，请先关闭旧服务）")
        return
    log(f"   启动 API（rerank_backend={backend}）...")
    env = {**os.environ, "PYTHONPATH": str(PROJECT), "PYTHONIOENCODING": "utf-8",
           "RERANK_BACKEND": backend}
    subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(PROJECT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    wait_for(lambda: http_ok(f"{API}/health"), "API", timeout=90)
    log("   API 就绪")


def warmup(backend: str) -> None:
    log("[4/5] 预热（本地精排首次要加载模型，约 30 秒）...")
    try:
        import httpx

        t0 = time.time()
        r = httpx.post(f"{API}/demo/ask", json={
            "question": "预热请求", "token": "demo-it-token", "with_answer": False,
        }, timeout=180)
        log(f"   预热完成（{time.time() - t0:.0f}s）")
    except Exception as e:  # noqa: BLE001
        log(f"   预热跳过：{e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("RERANK_BACKEND", "local"),
                    choices=["local", "llm"], help="精排后端")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    log("=" * 52)
    log("  RAG 演示 · 一键启动")
    log("=" * 52)
    ensure_docker()
    ensure_ollama()
    ensure_api(args.backend)
    warmup(args.backend)

    log("[5/5] 打开页面 ...")
    if not args.no_browser:
        webbrowser.open(DEMO_URL)
    log("")
    log(f"  ✅ 就绪：{DEMO_URL}")
    log("  （关闭服务：任务管理器结束 python 进程，或运行 scripts/stop_demo.py）")


if __name__ == "__main__":
    main()
