"""问答相关 DTO。"""
from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    departments: list[str] | None = None   # 模拟当前用户部门
    secret_level: int | None = None        # 模拟当前用户密级


class Source(BaseModel):
    source_file: str | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
