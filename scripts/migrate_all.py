"""对两个知识库各跑一次 alembic 迁移，并校验两库 schema 一致。

**为什么必须有两步**（缺一不可）：

1. **迁移要跑两次** —— 两库分离后，迁移只跑一个 → 两库 schema 不一致。
2. **迁移后要校验** —— 因为不一致**不会报错**：查询只会莫名失败，或列缺失时行为异常。
   这正是本项目反复踩的"**静默故障**"。所以逐列 + 逐索引比对，不一致就非零退出。

用法：
    python scripts/migrate_all.py             # 迁移两个库 + 校验
    python scripts/migrate_all.py --check     # 只校验，不迁移
    python scripts/migrate_all.py --kb real   # 只处理真实库

前置：目标库要能连上。压测库需先起容器（日常不启动，见 docker-compose profiles）：
    docker compose --profile stress up -d
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool, get_pool  # noqa: E402
from src.db.kb import KB_REAL, KB_STRESS, VALID_KBS  # noqa: E402

TABLES = ("chunks", "documents")


def _mask(url: str) -> str:
    """连接串打码后再打印（避免日志里出现口令）。"""
    return url.rsplit("@", 1)[-1] if "@" in url else url


def _migrate(kb: str) -> bool:
    url = get_settings().db_url(kb)
    print(f"[迁移] kb={kb} → {_mask(url)}", flush=True)
    env = {**os.environ, "ALEMBIC_DATABASE_URL": url}
    r = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=str(ROOT), env=env
    )
    ok = r.returncode == 0
    print(f"[迁移] kb={kb} {'✅ 成功' if ok else '❌ 失败'}\n", flush=True)
    return ok


async def _columns(table: str, kb: str) -> list[tuple[str, str]]:
    pool = await get_pool(kb)
    rows = await pool.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name=$1 ORDER BY column_name",
        table,
    )
    return [(r["column_name"], r["data_type"]) for r in rows]


async def _indexes(table: str, kb: str) -> list[str]:
    pool = await get_pool(kb)
    rows = await pool.fetch(
        "SELECT indexname FROM pg_indexes WHERE schemaname='public' AND tablename=$1 ORDER BY indexname",
        table,
    )
    return [r["indexname"] for r in rows]


async def _check() -> bool:
    print("[校验] 逐表比对两库 schema ...", flush=True)
    ok = True
    for table in TABLES:
        try:
            cols_a, cols_b = await _columns(table, KB_REAL), await _columns(table, KB_STRESS)
            idx_a, idx_b = await _indexes(table, KB_REAL), await _indexes(table, KB_STRESS)
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {table}: 连不上库（{e}）")
            print("     → 压测库没起？试：docker compose --profile stress up -d")
            return False

        if cols_a == cols_b:
            print(f"  ✅ {table} 列一致（{len(cols_a)} 列）")
        else:
            ok = False
            sa, sb = {c for c, _ in cols_a}, {c for c, _ in cols_b}
            print(f"  ❌ {table} 列不一致：")
            if sa - sb:
                print(f"       仅真实库有: {sorted(sa - sb)}")
            if sb - sa:
                print(f"       仅压测库有: {sorted(sb - sa)}")
            if sa == sb:
                print(f"       列名相同但类型不同")
                for c in sorted(sa):
                    ta = dict(cols_a).get(c)
                    tb = dict(cols_b).get(c)
                    if ta != tb:
                        print(f"         {c}: 真实={ta} / 压测={tb}")

        if idx_a == idx_b:
            print(f"  ✅ {table} 索引一致（{len(idx_a)} 个）")
        else:
            ok = False
            print(f"  ❌ {table} 索引不一致：仅真实{sorted(set(idx_a) - set(idx_b))} "
                  f"仅压测{sorted(set(idx_b) - set(idx_a))}")
    return ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不迁移")
    ap.add_argument("--kb", choices=[*VALID_KBS, "both"], default="both")
    args = ap.parse_args()

    kbs = list(VALID_KBS) if args.kb == "both" else [args.kb]

    if not args.check:
        results = {kb: _migrate(kb) for kb in kbs}
        if not all(results.values()):
            print("❌ 有库迁移失败，跳过校验")
            await close_pool()
            return 1

    ok = await _check()
    await close_pool()
    print("\n" + ("✅ 两库 schema 一致" if ok else "❌ 两库 schema 不一致 —— 有库漏跑迁移！"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
