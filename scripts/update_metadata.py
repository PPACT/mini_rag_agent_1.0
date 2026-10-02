"""`2.0-33` 元数据**独立更新路径** —— 改元数据，**不碰向量**。

## 为什么要有它

元数据（部门 / 密级 / 标题 / 弃用标记）本来就是**普通列**，直接 `UPDATE` 就行。
缺的**不是能力，是通道**：没有一条**受控**的路，就只能靠手写 SQL ——
而手写 SQL 里**最贵的错**是把 `content` / `embedding` 一起改了（→ 整块作废、要重嵌，
而且**不报错**）。

**本脚本的全部价值就是那条白名单**：

```
documents 可改：content_status
chunks    可改：department / secret_level / title / is_deprecated

⛔ content / embedding / content_tsv / content_hash / raw_table —— 一律拒绝
```

⚠️ **"不碰向量"是结构性保证，不是承诺**：SQL 由白名单**生成**，
`embedding` **根本进不了 SET 子句** —— 不存在"我小心一点就行"。
（`tests/test_update_metadata.py` 用**反向用例**钉住：喂一个禁止字段必须报错。）

⚠️ **改完必须失效缓存**：部门/密级一变，**谁能看到它**就变了 ——
不失效的话，旧权限下的答案会被**继续命中**（= 越权）。

用法：
    # 先看要改成什么（不改库）
    python scripts/update_metadata.py --kb stress --doc-id <uuid> --set department=HR --dry-run
    # 按文件名定位（重名会报错，不猜）
    python scripts/update_metadata.py --kb stress --filename 13_绩效考核实施指引.txt --set secret_level=2
    # 文档级的（content_status 在 documents 上）
    python scripts/update_metadata.py --kb stress --doc-id <uuid> --doc-set content_status=stale
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache.redis_client import invalidate_cache  # noqa: E402
from src.db.connection import close_pool, get_pool  # noqa: E402

# ⚠️ 白名单 = 本脚本的**全部安全边界**。加列之前先问：改它要重嵌吗？
_CHUNK_FIELDS: dict[str, type] = {
    "department": str,
    "secret_level": int,
    "title": str,
    "is_deprecated": bool,
}
_DOC_FIELDS: dict[str, type] = {
    "content_status": str,        # active / stale / deprecated（值由 DB 的 CHECK 兜底）
}

# ⛔ 显式列出来，是为了让"为什么不行"能报出来，而不是静默拒绝
_FORBIDDEN = {
    "content", "embedding", "content_tsv", "content_hash", "raw_table",
    "document_id", "chunk_index", "document_version",
}


def parse_set(items: list[str]) -> dict:
    """把 `--set k=v` 解析成 dict。⚠️ 值先按**字符串**收，类型由白名单定。"""
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set 要写成 key=value，收到的是 {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v
    return out


def coerce(fields: dict[str, str], allowed: dict[str, type]) -> dict:
    """按白名单**校验 + 转类型** —— 在这里挡住禁止字段与非法的值。

    ⚠️ **必须在校验失败时抛错**，不能跳过那一项继续跑：
    少改一个字段比报错更难发现。
    """
    bad = [k for k in fields if k in _FORBIDDEN]
    if bad:
        raise ValueError(
            f"这些字段**不属于元数据**、改它们会让块作废：{bad}。"
            f"（要改内容请走重新灌库，不要走这条路）"
        )
    unknown = [k for k in fields if k not in allowed]
    if unknown:
        raise ValueError(f"未知字段 {unknown}；可改的是 {sorted(allowed)}")
    out = {}
    for k, v in fields.items():
        t = allowed[k]
        if t is bool:
            if v.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{k} 要是布尔（true/false），收到 {v!r}")
            out[k] = v.lower() in ("true", "1")
        else:
            out[k] = t(v)
    return out


def build_update(table: str, fields: dict) -> tuple[str, list]:
    """生成 UPDATE 语句。

    ⭐ **本函数就是"不碰向量"的机械保证**：SET 子句**只由 `fields` 的键拼出**，
    而调用方必须先过 `coerce`（白名单）—— `embedding` 没有任何进入 SET 的路径。
    """
    if not fields:
        raise ValueError("没有要改的字段")
    cols = sorted(fields)
    sets = ", ".join(f"{c} = ${i + 2}" for i, c in enumerate(cols))
    # ⚠️ WHERE 的列名**必须写出来** —— 只写 `WHERE $1::uuid` 是**跑不通的 SQL**
    # （测试逮到过：`test_build_update_only_names_whitelisted_columns`）
    sql = f"UPDATE {table} SET {sets} WHERE {_KEY_COL[table]} = $1::uuid"   # noqa: S608
    return sql, [fields[c] for c in cols]


async def resolve_doc_id(pool, doc_id: str | None, filename: str | None) -> str:
    """定位文档。⚠️ 按文件名时**重名就报错** —— 不猜哪一个。"""
    if doc_id:
        return doc_id
    rows = await pool.fetch(
        "SELECT id FROM documents WHERE filename = $1 OR original_name = $1", filename
    )
    if len(rows) != 1:
        raise ValueError(f"按 {filename!r} 匹配到 {len(rows)} 份文档；请改用 --doc-id")
    return str(rows[0]["id"])


# 表 → 主键列（`chunks` 靠 document_id 关联到文档）
_KEY_COL = {"chunks": "document_id", "documents": "id"}


async def _snapshot(pool, table: str, doc_id: str, cols: list[str]) -> dict | None:
    """改前/改后快照 —— 用来**证明"只动了这几列"**，而不是嘴上说。

    `chunks` 是多行，取一行代表即可（同一文档的这几列本来就该一致）。
    """
    row = await pool.fetchrow(
        f"SELECT {', '.join(cols)} FROM {table} "      # noqa: S608 —— 表名/列名都来自内部常量与白名单
        f"WHERE {_KEY_COL[table]} = $1::uuid LIMIT 1",
        doc_id,
    )
    return dict(row) if row else None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True, help="real / stress")
    ap.add_argument("--doc-id", default=None)
    ap.add_argument("--filename", default=None, help="按文件名定位（重名会报错）")
    ap.add_argument("--set", action="append", default=[], help="chunks 的元数据，如 department=HR")
    ap.add_argument("--doc-set", action="append", default=[], help="documents 的元数据，如 content_status=stale")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行什么，不改库")
    args = ap.parse_args()

    if not args.set and not args.doc_set:
        print("[X] 至少给一个 --set 或 --doc-set")
        return 1
    if not args.doc_id and not args.filename:
        print("[X] 要给 --doc-id 或 --filename")
        return 1

    from src.db.kb import validate
    validate(args.kb)

    try:
        chunk_fields = coerce(parse_set(args.set), _CHUNK_FIELDS)
        doc_fields = coerce(parse_set(args.doc_set), _DOC_FIELDS)
    except ValueError as e:
        print(f"[X] {e}")
        return 1

    pool = await get_pool(args.kb)
    try:
        doc_id = await resolve_doc_id(pool, args.doc_id, args.filename)
        print(f"目标文档 {doc_id}（kb={args.kb}）")

        for table, fields in (("chunks", chunk_fields), ("documents", doc_fields)):
            if not fields:
                continue
            sql, params = build_update(table, fields)
            if args.dry_run:
                print(f"  [dry-run] {sql}  params={params}")
                continue
            cols = sorted(fields)
            before = await _snapshot(pool, table, doc_id, cols)
            status = await pool.execute(sql, doc_id, *params)   # ⚠️ SET 只含白名单列
            after = await _snapshot(pool, table, doc_id, cols)
            print(f"  {table}: {status}  {before} → {after}")

        if not args.dry_run:
            # ⚠️ 部门/密级一变，**谁能看到它**就变了 → 不失效 = 旧权限的答案继续被命中
            await invalidate_cache(args.kb)
            print(f"  ✅ 已失效 kb={args.kb} 的问答缓存（{args.kb}）")
        return 0
    finally:
        await close_pool()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
