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
from src.document_parser.chunking import split_blocks  # noqa: E402
from src.document_parser.loader import load_document  # noqa: E402
from src.document_parser.routing import DocumentParseError, failure_report  # noqa: E402
from src.embedding.base import get_embedding  # noqa: E402
from src.provenance import document_stamps, sha256_text  # noqa: E402
from src.vector_store.base import Chunk, get_vector_store  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
SYNTH_DIR = Path(__file__).resolve().parent / "corpus_synth"
MANIFEST = SYNTH_DIR / "_manifest.json"
COMPANY_SCOPE = "公司"     # 基础文档与通用文档：全员可见
EVAL_SECRET_LEVEL = 3

# 认识的扩展名。默认那两个目录里只有 .md/.txt，所以**加长这份清单不改变默认行为**；
# 它存在的意义是 `--dir docs/corpus` 时能收真实语料（pdf/docx/xlsx/...）。
# ⚠️ `.html` 会**显式失败**（`2.0-40` 未接入）—— 这不藏，让它进失败清单。
_SUFFIXES = (".md", ".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".py", ".html")


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


def collect_files(exclude_dept: bool, dirs: list[Path] | None = None) -> list[Path]:
    """收集语料文件。`exclude_dept=True` 时剔除部门变体（得到「干净语料」）。

    `dirs` 不给 = 默认那两份合成语料目录（**行为不变**）；
    给了就用它 —— 例如 `--dir docs/corpus` 灌真实语料（`2.0-47`）。
    """
    files: list[Path] = []
    for d in (dirs if dirs is not None else [CORPUS_DIR, SYNTH_DIR]):
        if d.exists():
            files.extend(sorted(p for p in d.glob("*") if p.suffix.lower() in _SUFFIXES))
    if exclude_dept:
        # 只剔除"部门变体"（文件名以 dept 开头），保留 LLM 生成的不同主题文档
        files = [p for p in files if not p.name.startswith("dept")]
    return files


async def truncate(kb: str = KB_STRESS) -> None:
    """清空评测数据（documents / chunks）。"""
    pool = await get_pool(kb)
    await pool.execute("TRUNCATE documents, chunks CASCADE")


async def ingest(exclude_dept: bool = False, dirs: list[Path] | None = None,
                 kb: str = KB_STRESS,
                 failures_out: list[DocumentParseError] | None = None) -> int:
    """灌入语料，返回切片总数。供 CLI 与评测脚本复用。

    ⚠️ `dirs` / `kb` / `failures_out` 都有默认值 —— **不传就是原来的行为**。

    `failures_out`（`2.0-50` 加的）：失败清单本来是**只在函数内部打印**的，
    调用方拿不到 —— 而 `2.0-50` 要的是"**哪一篇、什么原因**"。
    ⚠️ 用法是**传一个 list 进来**（不是改返回值）—— 免得打断既有调用方。
    """
    settings = get_settings()
    pool = await get_pool(kb)
    store = get_vector_store(kb)
    embedding = get_embedding()

    files = collect_files(exclude_dept, dirs)
    manifest = _load_manifest()
    src_desc = ("目录 " + ", ".join(str(d) for d in dirs)) if dirs is not None else "默认（基础 + 合成）"
    print(f"发现 {len(files)} 篇语料（{src_desc}"
          f"{'，已剔除部门变体' if exclude_dept else ''}），kb={kb}，清单条目 {len(manifest)}")

    total_chunks = 0
    scope_count: dict[str, int] = {}
    failures: list[DocumentParseError] = []   # 2.0-3：**失败文件清单**，不再静默
    for i, path in enumerate(files, 1):
        filename = path.name
        doc_id = doc_id_for(filename)
        scope = _scope_of(filename, manifest)   # 按真实范围标注（而非全标 IT）

        # ⚠️ **逐篇 try**：以前一篇解析失败会**中断整批灌库**（后面的文件全没进）。
        # 现在单篇失败只记进清单、继续跑 —— 最后统一打印。
        try:
            parsed = load_document(str(path))
            chunks = split_blocks(parsed, settings.chunk_size, settings.chunk_overlap)
            if not chunks:
                raise ValueError(f"解析后无有效文本（解析器 {parsed.parser}）")
            # ⚠️ 向量只算 content（表格是自然语言版）；原表随 raw_table 入库、只进词法
            vectors = await embedding.embed([c.text for c in chunks])
        except (DocumentParseError, ValueError) as e:
            failures.append(e if isinstance(e, DocumentParseError) else DocumentParseError(str(path), str(e)))
            if i % 100 == 0 or i == len(files):
                print(f"  进度 {i}/{len(files)} 篇，累计 {total_chunks} 切片（失败 {len(failures)}）")
            continue

        scope_count[scope] = scope_count.get(scope, 0) + 1
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
                page=c.page,
                raw_table=c.raw_table,
                table_complex=c.table_complex,
                # `2.0-30`：与生产入库路径**同一处**算（`provenance.sha256_text`）
                content_hash=sha256_text(c.text),
            )
            for j, c in enumerate(chunks)
        ]

        # 这条路径 filename 本来就是真文件名，故 original_name 与它一致 ——
        # 让「上传」与「评测灌库」两条入库路径语义对齐（D9-③）。
        # `2.0-30`：版本戳与生产路径**同一处**算（`provenance.document_stamps`）
        stamps = document_stamps(path)
        await pool.execute(
            """
            INSERT INTO documents (id, filename, original_name, status, chunk_count,
                                   source_hash, parser_version, clean_rules_version,
                                   chunker_version, embedding_model)
            VALUES ($1::uuid, $2, $2, 'completed', $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE
              SET filename = EXCLUDED.filename,
                  original_name = EXCLUDED.original_name,
                  status = 'completed',
                  chunk_count = EXCLUDED.chunk_count,
                  source_hash = EXCLUDED.source_hash,
                  parser_version = EXCLUDED.parser_version,
                  clean_rules_version = EXCLUDED.clean_rules_version,
                  chunker_version = EXCLUDED.chunker_version,
                  embedding_model = EXCLUDED.embedding_model,
                  is_deleted = false,
                  updated_at = now()
            """,
            doc_id, filename, len(chunks),
            stamps["source_hash"], stamps["parser_version"],
            stamps["clean_rules_version"], stamps["chunker_version"],
            stamps["embedding_model"],
        )
        await store.replace_document(doc_id, chunk_objs, vectors)
        total_chunks += len(chunks)

        if i % 100 == 0 or i == len(files):
            print(f"  进度 {i}/{len(files)} 篇，累计 {total_chunks} 切片（失败 {len(failures)}）")

    top = sorted(scope_count.items(), key=lambda kv: -kv[1])[:5]
    print(f"范围分布（前5）: {top}{' ...' if len(scope_count) > 5 else ''}")

    # 2.0-3：失败清单**一定要打出来**（以前只会在中途炸掉，看不到哪几篇坏了）
    if failures_out is not None:
        failures_out.extend(failures)
    if failures:
        print(f"\n⚠️ {len(failures)} 篇解析失败（已跳过，未入库）：")
        for err in failures[:20]:
            print(failure_report(err))
        if len(failures) > 20:
            print(f"  …另有 {len(failures) - 20} 篇，见上方规律")
    else:
        print("\n✅ 全部解析成功，无失败文件")
    total = await store.count()
    print(f"完成，库中切片总数 {total}")
    return total


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dept", action="store_true",
                        help="不灌部门变体（得到「干净语料」，用于日常集评估）")
    parser.add_argument("--truncate", action="store_true", help="灌之前先清空")
    parser.add_argument("--dir", action="append", default=None,
                        help="语料目录（可重复）。不给 = 默认那两份合成语料目录")
    parser.add_argument("--kb", default=KB_STRESS, help=f"目标库（默认 {KB_STRESS}）")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dir] if args.dir else None
    if args.truncate:
        await truncate(args.kb)
        print(f"已清空旧数据（kb={args.kb}）")
    await ingest(args.exclude_dept, dirs, args.kb)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
