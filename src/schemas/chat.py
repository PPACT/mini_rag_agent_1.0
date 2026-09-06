"""问答相关 DTO。"""
from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    # 注意：department / secret_level 不再由客户端传入，改由服务端从鉴权 token 解析。


class Source(BaseModel):
    source_file: str | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
