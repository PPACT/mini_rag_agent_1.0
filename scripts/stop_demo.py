"""停止演示服务（按端口找进程，只杀本项目占用的 8000 端口）。

用法：python scripts/stop_demo.py
（Docker 容器与 Ollama 不停——它们可复用；要停用：docker compose down）
"""
from __future__ import annotations

import subprocess
import sys


def _pids_on_port(port: int) -> set[int]:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    pids = set()
    for line in out.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.add(int(parts[-1]))
    return pids


def main() -> None:
    pids = _pids_on_port(8000)
    if not pids:
        print("8000 端口没有监听进程（API 未运行）")
        return
    for pid in pids:
        r = subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(f"  已停止 PID {pid}" if r.returncode == 0 else f"  停止 {pid} 失败: {r.stderr.strip()}")
    print("API 已停止（Docker 容器与 Ollama 仍在运行）")


if __name__ == "__main__":
    main()
