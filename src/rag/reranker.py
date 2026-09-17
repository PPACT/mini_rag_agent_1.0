"""两段式检索的精排层：对粗排召回的候选按相关性重排。

背景：向量检索是"粗排"，常把语义相近但答非所问的片段排在前面。
精排（Rerank）用更强的判断力纠正排序——这是当前检索质量的第一杠杆。
"""
from __future__ import annotations

import asyncio
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
                ],
                temperature=0,  # 精排必须可复现：同一输入应给出同一排序
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


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class LocalCrossEncoderReranker(BaseReranker):
    """本地 cross-encoder 精排（默认 bge-reranker-base）。

    与 LLM 精排的关键差异：
    - **确定性**：同输入同输出（无温度随机、无网络波动）
    - **快**：GPU 上 20 条候选约数十毫秒（LLM 实测约 9 秒）
    - 零 API 成本

    代价：首次调用时懒加载模型（~1.1GB），常驻显存约 1.2GB（见 `docs/资源规划.md`）。
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        s = get_settings()
        self._model_name = model_name or s.rerank_local_model
        self._device = device or s.rerank_local_device or ("cuda" if _cuda_available() else "cpu")
        self._snippet = s.rerank_snippet_chars
        self._model = None

    def _ensure_model(self):
        """懒加载（首次调用时才载入权重）。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
            audit("rerank_model_loaded", model=self._model_name, device=self._device)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if len(chunks) <= top_n:
            return chunks[:top_n]
        model = self._ensure_model()
        pairs = [(query, c.content[: self._snippet]) for c in chunks]
        # CrossEncoder.predict 是同步阻塞的 → 丢到线程池，避免卡住事件循环
        scores = await asyncio.to_thread(model.predict, pairs)
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        return [chunks[i] for i in order][:top_n]


_reranker_instance: BaseReranker | None = None


def get_reranker() -> BaseReranker:
    """按配置返回精排实现（**单例**）：rerank_backend = 'llm'（默认）| 'local'。

    必须是单例——本地模型加载一次要 160+ 秒，每次新建实例会导致每个请求都重载模型。
    """
    global _reranker_instance
    if _reranker_instance is None:
        if get_settings().rerank_backend == "local":
            _reranker_instance = LocalCrossEncoderReranker()
        else:
            _reranker_instance = LLMReranker()
    return _reranker_instance
