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
    # 判定三态（P0-1）：clear（有答案）｜ ambiguous（无唯一答案）｜ unknown（无法判断）
    #
    # ⚠️ **为什么不复用 `need_clarification`**：那个字段的契约是
    #    「为 True 时 answer **是澄清话术**」；而 `unknown` 时 answer 是**真实答案**
    #    （前面带不确定标注、末尾带一句反问）。拿它表达会破坏契约，前端会误判。
    #    故三态单独一个字段，`need_clarification` 语义保持不变。
    judge_status: str = "clear"
