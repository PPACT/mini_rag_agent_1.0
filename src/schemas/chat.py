"""问答相关 DTO。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    # 注意：department / secret_level 不再由客户端传入，改由服务端从鉴权 token 解析。


class Source(BaseModel):
    source_file: str | None
    chunk_index: int
    score: float


class ClarifyOption(BaseModel):
    """歧义澄清时给出的候选（让用户明确意图）。"""

    source: str
    summary: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    # 歧义处理：为 True 时 answer 是澄清话术，clarify_options 列出候选
    need_clarification: bool = False
    clarify_options: list[ClarifyOption] = Field(default_factory=list)
