"""文档上传接口。"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.config.settings import get_settings
from src.db.connection import get_pool
from src.schemas.document import DocumentStatus, UploadResponse
from src.tasks.queue import enqueue_process_document

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED = {"pdf", "docx", "doc", "pptx", "ppt"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: str | None = Form(None),
    secret_level: int = Form(0),
) -> UploadResponse:
    """上传文档：落盘 -> 写 documents(pending) -> 入队异步处理（附权限元数据）。"""
    settings = get_settings()
    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in (file.filename or "") else ""
    if ext not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    os.makedirs(settings.upload_dir_abs, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{ext}"
    dest = os.path.join(settings.upload_dir_abs, filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    pool = await get_pool()
    doc_id = await pool.fetchval(
        "INSERT INTO documents (filename, status) VALUES ($1, 'pending') RETURNING id::text", filename
    )
    await enqueue_process_document(doc_id, department, secret_level)
    return UploadResponse(document_id=doc_id, status="pending")


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document_status(document_id: str) -> DocumentStatus:
    """查询文档处理状态。"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id::text AS document_id, filename, status, chunk_count, error FROM documents WHERE id = $1::uuid",
        document_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentStatus(**dict(row))
