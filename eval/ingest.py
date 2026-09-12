"""把 eval/corpus 下的评测语料灌入向量库。

可重复执行（用文件名派生的确定性 UUID，重跑即覆盖）。
用法：python eval/ingest.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings  # noqa: E402
from src.db.connection import close_pool, get_pool  # noqa: E402
from src.document_parser.loader import load_text  # noqa: E402
from src.document_parser.semantic_splitter import split_text  # noqa: E402
from src.embedding.base import get_embedding  # noqa: E402
from src.vector_store.base import Chunk, get_vector_store  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
SYNTH_DIR = Path(__file__).resolve().parent / "corpus_synth"
EVAL_DEPARTMENT = "IT"
EVAL_SECRET_LEVEL = 3


def doc_id_for(filename: str) -> str:
    """由文件名派生确定性 UUID，保证重跑幂等。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"eval:{filename}"))


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dept", action="store_true", help="不灌部门变体（隔离规模 vs 近重复两个因素）")
    args = parser.parse_args()
    exclude_dept = args.exclude_dept

    settings = get_settings()
    pool = await get_pool()
    store = get_vector_store()
    embedding = get_embedding()

    files: list[Path] = []
    for d in (CORPUS_DIR, SYNTH_DIR):
        if d.exists():
            files.extend(sorted(p for p in d.glob("*") if p.suffix.lower() in (".md", ".txt")))
    if exclude_dept:
        # 只剔除"部门变体"（文件名以 dept 开头），保留 LLM 生成的不同主题文档
        files = [p for p in files if not p.name.startswith("dept")]
    print(f"发现 {len(files)} 篇语料{'（已剔除部门变体）' if exclude_dept else '（基础 + 合成）'}")

    total_chunks = 0
    for i, path in enumerate(files, 1):
        filename = path.name
        doc_id = doc_id_for(filename)

        text = load_text(str(path))
        chunks = split_text(text, settings.chunk_size, settings.chunk_overlap)
        vectors = await embedding.embed([c.text for c in chunks])

        chunk_objs = [
            Chunk(
                document_id=doc_id,
                chunk_index=j,
                content=c.text,
                source_file=filename,
                department=EVAL_DEPARTMENT,
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

    print(f"完成，库中切片总数 {await store.count()}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
