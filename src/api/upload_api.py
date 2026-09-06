"""文档上传接口。"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.auth.deps import User, get_current_user
from src.config.settings import get_settings
from src.db.connection import get_pool
from src.schemas.document import DocumentStatus, UploadResponse
from src.tasks.queue import enqueue_process_document

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED = {"pdf", "docx", "doc", "pptx", "ppt"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    secret_level: int = Form(0),
    user: User = Depends(get_current_user),
) -> UploadResponse:
    """上传文档：落盘 -> 写 documents(pending) -> 入队异步处理。

    department 取自鉴权 token（服务端身份）；secret_level 为文档密级（表单），
    校验不能超过上传者自己的密级。
    """
    settings = get_settings()
    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in (file.filename or "") else ""
    if ext not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    if secret_level > user.secret_level:
        raise HTTPException(
            status_code=403,
            detail=f"无权上传密级 {secret_level} 的文档（你的密级为 {user.secret_level}）",
        )

    os.makedirs(settings.upload_dir_abs, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{ext}"
    dest = os.path.join(settings.upload_dir_abs, filename)
    with open(dest, "wb") as f:
        f.write(await file.read())

    pool = await get_pool()
    doc_id = await pool.fetchval(
        "INSERT INTO documents (filename, status) VALUES ($1, 'pending') RETURNING id::text", filename
    )
    await enqueue_process_document(doc_id, user.department, secret_level)
    return UploadResponse(document_id=doc_id, status="pending")


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document_status(
    document_id: str,
    user: User = Depends(get_current_user),
) -> DocumentStatus:
    """查询文档处理状态（需鉴权）。"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id::text AS document_id, filename, status, chunk_count, error FROM documents WHERE id = $1::uuid",
        document_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentStatus(**dict(row))
