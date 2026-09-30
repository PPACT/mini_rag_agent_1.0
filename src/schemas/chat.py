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

    # ---- 2.0：引用**随行**信息（2.0-1 / 2.0-4）----
    page: int | None = None
    """页码（1-based）。**不适用就是 None** —— Word/Excel 无稳定页码，不拿 0 假装。"""

    raw_table: str | None = None
    """表格块的**原表 Markdown**，随引用一起给前端（2.0-1 口径）。

    ⚠️ 为什么这里**带**、而送 LLM 的上下文**不带**：
    - 给**人**：原表的价值最大 —— 用户能**自己核对数值**（这正是"保留原表"的初衷）；
    - 给**LLM**：自然语言版已被"数值保真"这条验收管住，再塞一遍只占 token、且与 content 重复。
    两条去向的理由不同，别把它们当成同一条。
    """


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
