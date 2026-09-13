"""两段式检索的精排层：对粗排召回的候选按相关性重排。

背景：向量检索是"粗排"，常把语义相近但答非所问的片段排在前面。
精排（Rerank）用更强的判断力纠正排序——这是当前检索质量的第一杠杆。
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from src.config.litellm_client import complete
from src.config.prompts import load_rerank_templates
from src.config.settings import get_settings
from src.observability.tracer import audit
from src.vector_store.base import Chunk

_JSON_ARRAY = re.compile(r"\[[\d,\s]*\]")


class BaseReranker(ABC):
    """精排统一接口。上层只依赖此接口，切换实现（LLM / 本地模型 / 云端 API）不改业务代码。"""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """按与 query 的相关性重排 chunks，返回前 top_n 条。"""
        raise NotImplementedError


class LLMReranker(BaseReranker):
    """用 LLM 对候选排序。零显存，代价是多一次调用（延迟 + token）。"""

    def __init__(self) -> None:
        self._snippet = get_settings().rerank_snippet_chars

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if len(chunks) <= top_n:
            return chunks[:top_n]

        system_tpl, human_tpl = load_rerank_templates()
        candidates = "\n".join(
            f"[{i}] {c.content[: self._snippet]}" for i, c in enumerate(chunks)
        )
        try:
            raw = await complete(
                [
                    {"role": "system", "content": system_tpl},
                    {"role": "user", "content": human_tpl.format(question=query, candidates=candidates)},
                ]
            )
            order = self._parse(raw, len(chunks))
            audit("rerank", query=query, candidates=len(chunks), reordered=bool(order))
        except Exception as e:  # noqa: BLE001
            # 精排是"锦上添花"，失败不能拖垮检索主链路 → 回退到粗排顺序
            audit("rerank_failed", query=query, error=str(e))
            order = []

        if not order:
            return chunks[:top_n]
        return [chunks[i] for i in order][:top_n]

    @staticmethod
    def _parse(raw: str, n: int) -> list[int]:
        """解析 LLM 输出的编号序列；补全漏掉的编号（保序）；无法解析返回空。"""
        match = _JSON_ARRAY.search(raw or "")
        if not match:
            return []
        try:
            arr = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        ordered: list[int] = []
        for x in arr:
            if isinstance(x, int) and 0 <= x < n and x not in ordered:
                ordered.append(x)
        ordered.extend(i for i in range(n) if i not in ordered)
        return ordered


class LocalCrossEncoderReranker(BaseReranker):
    """占位：本地 cross-encoder（如 bge-reranker-base）。

    显存约束见 `docs/资源规划.md`：8GB 机器上选 base 版（~1.2GB），且须支持 CPU 降级。
    """

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        raise NotImplementedError("本地 cross-encoder 尚未接入，仅占位")


def get_reranker() -> BaseReranker:
    """返回当前精排实现（默认 LLM；后期可切本地 cross-encoder）。"""
    return LLMReranker()
