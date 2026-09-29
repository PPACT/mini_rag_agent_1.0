"""端到端验证：两库分离后各入口是否正确路由（`eval/` 下，已纳入版本库——**证据脚本不能放会被清理的 `logs/`**）。

验证 4 件事：
  ① /demo/ask kb=stress → 有命中（合成语料在压测库）
  ② /demo/ask kb=real   → **0 命中且不回退**（真实库是空的）
  ③ /demo/ask kb=非法值 → **400**（fail-closed，不静默回退到某个库）
  ④ /chat              → 走**真实库**（空）→ 不应出现合成语料来源（串库检测）
"""
import httpx

BASE = "http://127.0.0.1:8000"
Q = "我迟到了会怎么样？"
SYNTH = {"hr_attendance.md", "dept00_hr_attendance.md", "it_leave.md"}


def demo(kb: str, **kw) -> httpx.Response:
    return httpx.post(
        f"{BASE}/demo/ask",
        json={"question": Q, "token": "demo-it-token", "kb": kb, **kw},
        timeout=300,
    )


def main() -> int:
    ok = True

    print("① /demo/ask kb=stress （应命中）")
    d = demo("stress", with_answer=False).json()
    final = d.get("final", [])
    print(f"   kb字段={d.get('kb')}  命中={len(final)}  来源={sorted({c['source'] for c in final})[:3]}")
    ok &= d.get("kb") == "stress" and len(final) > 0

    print("② /demo/ask kb=real （真实库为空 → 应 0 命中）")
    d = demo("real", with_answer=False).json()
    final = d.get("final", [])
    print(f"   kb字段={d.get('kb')}  命中={len(final)}  来源={sorted({c['source'] for c in final})}")
    ok &= d.get("kb") == "real" and len(final) == 0

    print("③ /demo/ask kb=bogus （应 400，fail-closed）")
    r = demo("bogus")
    print(f"   HTTP {r.status_code}  detail={str(r.json().get('detail', ''))[:70]}")
    ok &= r.status_code == 400

    print("④ /chat （Bearer，应走真实库 → 不得出现合成语料来源）")
    r = httpx.post(
        f"{BASE}/chat", json={"question": Q},
        headers={"Authorization": "Bearer demo-it-token"}, timeout=300,
    )
    d = r.json()
    srcs = {s.get("source_file") for s in d.get("sources", []) if s.get("source_file")}
    leaked = srcs & SYNTH
    print(f"   HTTP {r.status_code}  来源={sorted(srcs)}")
    print(f"   串到压测库了吗: {'❌ 串了！' if leaked else '✅ 没有'}")
    ok &= not leaked

    print("\n" + ("✅ 端到端验证全部通过" if ok else "❌ 有项未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
