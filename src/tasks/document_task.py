"""文档解析 -> 切块 -> 向量化 -> 入库 的异步任务。"""
from __future__ import annotations

import os

from src.cache.redis_client import invalidate_cache
from src.config.settings import get_settings
from src.db.connection import get_pool
from src.document_parser.chunking import split_blocks
from src.document_parser.loader import load_document
from src.embedding.base import get_embedding
from src.observability.tracer import audit
from src.provenance import document_stamps, sha256_text
from src.vector_store.base import Chunk, get_vector_store


async def process_document(
    ctx: dict, kb: str, document_id: str, department: str | None = None, secret_level: int = 0
) -> None:
    """后台任务：解析 -> 结构化切块 -> 向量化 -> 原子写库 -> 更新状态。

    `kb` 由入队时确定（真实 / 压测），决定**读文档状态与写切片**都落在哪个库。
    """
    settings = get_settings()
    pool = await get_pool(kb)

    row = await pool.fetchrow(
        "SELECT id, filename, original_name, version FROM documents WHERE id = $1::uuid",
        document_id,
    )
    if row is None:
        return

    await pool.execute(
        "UPDATE documents SET status='processing', updated_at=now() WHERE id=$1::uuid", document_id
    )
    try:
        file_path = os.path.join(settings.upload_dir_abs, row["filename"])
        # 2.0：走三层路由解析（返回块 + 页码 + 原表），不再只拿一份纯文本
        parsed = load_document(file_path)
        chunks = split_blocks(parsed, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            # 报错要说清"用了哪个解析器、解出多少块"——否则只是又一句"无有效文本"
            raise ValueError(
                f"解析后无有效文本（解析器 {parsed.parser}，"
                f"块数 {len(parsed.blocks)}，警告 {len(parsed.warnings)}）"
            )

        # 审计：**实际用了哪个解析器**、各类块多少（协议 P-2 要能核实实际生效值）
        audit("parse", document_id=document_id, kb=kb, parser=parsed.parser,
              blocks=len(parsed.blocks), kinds=parsed.kind_counts(),
              chunks=len(chunks), warnings=parsed.warnings[:5])

        embedding = get_embedding()
        # ⚠️ **向量只算 content**（表格这里是自然语言版）——
        # 原表**不进向量**，而是随 `raw_table` 入库、只进词法（2.0-1 口径）。
        # 裸表的语义依赖行列头，向量抓不住；且吃它会挤占向量 top-k 席位。
        vectors = await embedding.embed([c.text for c in chunks])

        version = row["version"]
        # D9-③：`source_file` 是**对外溯源**的字段（还会进 LLM 上下文），
        # 必须用原始文件名而非 UUID 落盘名。存量行 original_name 为 NULL
        # （0004 迁移不回填假数据），此时回落 filename。
        source_name = row["original_name"] or row["filename"]
        chunk_objs = [
            Chunk(
                document_id=document_id,
                chunk_index=i,
                content=c.text,
                source_file=source_name,
                document_version=version,
                department=department,
                secret_level=secret_level,
                start_offset=c.start,
                end_offset=c.end,
                title=c.title,
                page=c.page,
                raw_table=c.raw_table,
                table_complex=c.table_complex,
                # `2.0-30`：块文本哈希 —— 让"改口径后只重嵌变了的块"成为可能
                content_hash=sha256_text(c.text),
            )
            for i, c in enumerate(chunks)
        ]

        store = get_vector_store(kb)
        await store.replace_document(document_id, chunk_objs, vectors)

        # `2.0-30`：把**这次实际用的管道口径**落到文档行上。
        # ⚠️ 两条入库路径都调 `provenance.document_stamps`（一处定义，别各写一份）。
        # ⚠️ `content_status` **这里不动** —— 它是"还算不算数"的标记，
        #    重跑一份文档不代表它自动变成"当前口径该留的"（那要等 `2.0-5` 那批再定语义）。
        stamps = document_stamps(file_path)
        await pool.execute(
            """
            UPDATE documents
               SET status='completed', chunk_count=$2, updated_at=now(),
                   source_hash=$3, parser_version=$4, clean_rules_version=$5,
                   chunker_version=$6, embedding_model=$7
             WHERE id=$1::uuid
            """,
            document_id,
            len(chunks),
            stamps["source_hash"],
            stamps["parser_version"],
            stamps["clean_rules_version"],
            stamps["chunker_version"],
            stamps["embedding_model"],
        )
        await invalidate_cache(kb)   # D9-⑦：只清**本库**的问答缓存，不波及另一个库
    except Exception as e:  # noqa: BLE001
        await pool.execute(
            "UPDATE documents SET status='failed', error=$2, updated_at=now() WHERE id=$1::uuid",
            document_id,
            str(e),
        )
        raise  # 交给 arq 重试；超 max_tries 后 job 判失败，状态已标 failed（死信兜底）
