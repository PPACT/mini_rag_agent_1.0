"""文档相关 DTO。"""
from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    status: str = "pending"


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int = 0
    error: str | None = None
