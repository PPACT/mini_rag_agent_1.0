"""文档相关 DTO。"""
from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    status: str = "pending"


class DocumentStatus(BaseModel):
    document_id: str
    # filename = UUID 落盘名（内部）；对外展示请用 original_name（存量行可能为 None）
    filename: str
    original_name: str | None = None
    status: str
    chunk_count: int = 0
    error: str | None = None
