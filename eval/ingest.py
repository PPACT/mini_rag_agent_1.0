"""把 eval/corpus 下的评测语料灌入向量库。

可重复执行（用文件名派生的确定性 UUID，重跑即覆盖）。
用法：python eval/ingest.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool, get_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.document_parser.loader import load_text  # noqa: E402
from src.document_parser.semantic_splitter import split_text  # noqa: E402
from src.embedding.base import get_embedding  # noqa: E402
from src.vector_store.base import Chunk, get_vector_store  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
SYNTH_DIR = Path(__file__).resolve().parent / "corpus_synth"
MANIFEST = SYNTH_DIR / "_manifest.json"
COMPANY_SCOPE = "公司"     # 基础文档与通用文档：全员可见
EVAL_SECRET_LEVEL = 3


def _load_manifest() -> dict[str, str]:
    """读取「部门变体文件 → 所属部门」映射（由 gen_synth.py 生成）。"""
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def _scope_of(filename: str, manifest: dict[str, str]) -> str:
    """文件的范围：部门变体 → 其部门；其余（基础/通用文档）→ 公司级。"""
    return manifest.get(filename, COMPANY_SCOPE)


def doc_id_for(filename: str) -> str:
    """由文件名派生确定性 UUID，保证重跑幂等。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"eval:{filename}"))


def collect_files(exclude_dept: bool) -> list[Path]:
    """收集语料文件。exclude_dept=True 时剔除部门变体（得到「干净语料」）。"""
    files: list[Path] = []
    for d in (CORPUS_DIR, SYNTH_DIR):
        if d.exists():
            files.extend(sorted(p for p in d.glob("*") if p.suffix.lower() in (".md", ".txt")))
    if exclude_dept:
        # 只剔除"部门变体"（文件名以 dept 开头），保留 LLM 生成的不同主题文档
        files = [p for p in files if not p.name.startswith("dept")]
    return files


async def truncate() -> None:
    """清空评测数据（documents / chunks）。"""
    pool = await get_pool(KB_STRESS)
    await pool.execute("TRUNCATE documents, chunks CASCADE")


async def ingest(exclude_dept: bool = False) -> int:
    """灌入语料，返回切片总数。供 CLI 与评测脚本复用。"""
    settings = get_settings()
    pool = await get_pool(KB_STRESS)
    store = get_vector_store(KB_STRESS)
    embedding = get_embedding()

    files = collect_files(exclude_dept)
    manifest = _load_manifest()
    print(f"发现 {len(files)} 篇语料{'（已剔除部门变体）' if exclude_dept else '（基础 + 合成）'}"
          f"，清单条目 {len(manifest)}")

    total_chunks = 0
    scope_count: dict[str, int] = {}
    for i, path in enumerate(files, 1):
        filename = path.name
        doc_id = doc_id_for(filename)
        scope = _scope_of(filename, manifest)   # 按真实范围标注（而非全标 IT）
        scope_count[scope] = scope_count.get(scope, 0) + 1

        text = load_text(str(path))
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        vectors = await embedding.embed([c.text for c in chunks])

        chunk_objs = [
            Chunk(
                document_id=doc_id,
                chunk_index=j,
                content=c.text,
                source_file=filename,
                department=scope,
                secret_level=EVAL_SECRET_LEVEL,
                start_offset=c.start,
                end_offset=c.end,
                title=c.title,
            )
            for j, c in enumerate(chunks)
        ]

        await pool.execute(
            """
            INSERT INTO documents (id, filename, status, chunk_count)
            VALUES ($1::uuid, $2, 'completed', $3)
            ON CONFLICT (id) DO UPDATE
              SET filename = EXCLUDED.filename,
                  status = 'completed',
                  chunk_count = EXCLUDED.chunk_count,
                  is_deleted = false,
                  updated_at = now()
            """,
            doc_id, filename, len(chunks),
        )
        await store.replace_document(doc_id, chunk_objs, vectors)
        total_chunks += len(chunks)

        if i % 100 == 0 or i == len(files):
            print(f"  进度 {i}/{len(files)} 篇，累计 {total_chunks} 切片")

    top = sorted(scope_count.items(), key=lambda kv: -kv[1])[:5]
    print(f"范围分布（前5）: {top}{' ...' if len(scope_count) > 5 else ''}")
    total = await store.count()
    print(f"完成，库中切片总数 {total}")
    return total


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dept", action="store_true",
                        help="不灌部门变体（得到「干净语料」，用于日常集评估）")
    parser.add_argument("--truncate", action="store_true", help="灌之前先清空")
    args = parser.parse_args()

    if args.truncate:
        await truncate()
        print("已清空旧数据")
    await ingest(args.exclude_dept)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
