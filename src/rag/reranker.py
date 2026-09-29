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
            if not order:
                # ⚠️ 解析不到编号（如模型返回空）≠ "模型确认原顺序"。两者以前记录得一模一样，
                # 导致精排整段静默空转还看着正常（实测：推理模型思考吃满预算时必然发生）。
                audit("rerank_empty", query=query, candidates=len(chunks), raw_len=len(raw or ""))
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
    - **快**：整条检索链路 p50 —— 不精排 **1985ms** ／ 本地精排 **2120ms** ／ LLM 精排 **2682ms**
      （`优化方案.md` §8 矩阵实测，6 组 × 80 题，思考链关闭）
      ⚠️ 本注释曾写「LLM 实测约 9 秒」——那是**思考链 bug 期的数字，已作废**
      （见 `优化方案.md` §5；实测 LLM 精排只多约 **0.7s**）
    - 零 API 成本

    代价：首次调用时懒加载模型（~1.1GB），常驻显存约 1.2GB（见 `docs/local/资源规划.md`）。
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        s = get_settings()
        self._model_name = model_name or s.rerank_local_model
        self._device = device or s.rerank_local_device or ("cuda" if _cuda_available() else "cpu")
        self._snippet = s.rerank_snippet_chars
        self._model = None

    def _ensure_model(self):
        """懒加载（首次调用时才载入权重）。

        `local_files_only=True` 跳过 huggingface_hub 的联网"检查更新"——
        模型已在本地缓存时，这一步会白等 20-40 秒（网络慢时更久）。
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(
                    self._model_name, device=self._device, local_files_only=True,
                )
            except Exception:  # noqa: BLE001
                # 首次使用：本地无缓存 → 回退在线加载（会触发一次下载）
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


VALID_BACKENDS: tuple[str, ...] = ("local", "llm")

# ⚠️ **按 backend 分别持单例**（不是"一个全局单例"）——D11 的关键点：
#    原来只存一个实例、第一次调用就把后端固化了。若只加个请求级参数而不动这里，
#    **切了后端也仍用第一次那个** → 「UI 变了、行为不变、还不报错」，
#    正是本项目反复踩的那类"配置值不生效"陷阱（P-1 / P-2）。
#    local 与 llm 两个实例各自懒加载、可共存（切回来不必重载模型）。
_rerankers: dict[str, BaseReranker] = {}


def _make_reranker(backend: str) -> BaseReranker:
    if backend == "local":
        return LocalCrossEncoderReranker()
    return LLMReranker()


def resolve_backend(backend: str | None = None) -> str:
    """把「请求级后端（可为 None）」解析成**实际生效的**后端名，非法即抛错。

    ⚠️ 独立成一个函数，是为了让**回显**与**取实例**走同一套归一化逻辑 ——
    否则页面回显的和实际用的可能不是同一个值（P-2：要核实实际生效值）。
    """
    name = (backend if backend is not None else get_settings().rerank_backend or "")
    name = name.strip().lower()
    if name not in VALID_BACKENDS:
        raise ValueError(
            f"未知的精排后端 backend={backend!r}；合法值：{VALID_BACKENDS}"
            f"（None = 跟随配置 RERANK_BACKEND）"
        )
    return name


def get_reranker(backend: str | None = None) -> BaseReranker:
    """返回精排实现（**按 backend 分别持单例**）：`local`（默认）| `llm`。

    `backend=None` → 用 `settings.rerank_backend`（默认 `local`，取舍见 settings 注释）。
    显式传入则覆盖配置 —— 供 `/demo` 的后端选择器与离线对比评测使用。

    必须缓存实例（而非每次新建）——本地模型加载一次要 160+ 秒，
    每次请求新建会导致每个请求都重载模型。
    """
    name = resolve_backend(backend)
    if name not in _rerankers:
        _rerankers[name] = _make_reranker(name)
    return _rerankers[name]
