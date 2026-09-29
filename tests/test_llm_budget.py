"""D9-⑨⑩：LLM 的 token 预算 / 超时 / 重试必须**来自配置**，且两条路径**有意不同档**。

背景（本项目要求：**延迟不可接受**）：原来两条路的超时都是硬编码 60s，
生成侧还叠加 `max_retries=3` → **最坏 180s+**，而正常耗时只有 ~2.5s。
"对齐两条路径的重试次数"是**错的**解法（会把延迟敏感的那条也拖差），
正解是**各按正常耗时定档 + 给上界**。

本文件钉住三件事：
1. `complete()`（判定/精排）确实把配置值传给了 litellm；
2. **默认档位本身**是那组"有上界"的值（有人想放宽时会在这里红）；
3. 思考链**主动开启**时告警会响（`thinking_leaked` 只覆盖"泄漏"，不覆盖"主动开"）。
"""
from __future__ import annotations

import asyncio

from src.agent import graph_builder
from src.config import litellm_client
from src.config.settings import get_settings


class _StubSettings:
    """只提供被测代码用到的字段，避免依赖真实 .env（含密钥）。"""

    llm_api_key = "test-key"
    llm_provider = "deepseek"
    llm_model = "test-model"
    llm_base_url = "https://example.invalid"
    llm_thinking_enabled = False
    llm_max_tokens = 777
    llm_timeout_judge = 3.5
    llm_max_retries_judge = 2
    llm_timeout_generate = 11.0
    llm_max_retries_generate = 4


class _Msg:
    content = "ok"
    reasoning_content = None


class _Resp:
    choices = [type("_C", (), {"message": _Msg()})()]


def _capture(monkeypatch, **overrides) -> dict:
    """跑一次 complete()，返回实际传给 litellm.acompletion 的关键字参数。"""
    stub = _StubSettings()
    for k, v in overrides.items():
        setattr(stub, k, v)
    monkeypatch.setattr(litellm_client, "get_settings", lambda: stub)
    monkeypatch.setattr(litellm_client, "_thinking_warned", True)  # 静音告警，只看参数
    seen: dict = {}

    async def _fake_acompletion(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(litellm_client.litellm, "acompletion", _fake_acompletion)
    asyncio.run(litellm_client.complete([{"role": "user", "content": "hi"}]))
    return seen


def test_complete_takes_budget_from_settings(monkeypatch):
    """⑨⑩ 回归：max_tokens / timeout / num_retries 都不得再硬编码。"""
    kwargs = _capture(monkeypatch)
    assert kwargs["max_tokens"] == 777            # ⑩：原为硬编码 2048
    assert kwargs["timeout"] == 3.5               # ⑨：原为硬编码 60
    assert kwargs["num_retries"] == 2             # ⑨：显式传，不再吃 litellm 默认值


def test_default_budget_is_bounded():
    """默认档位是"有上界"的那组 —— 放宽它必须是一次**显式**改动。

    最坏耗时 = timeout × (1 + max_retries)：
      判定/精排 8s × (1+0) = 8s ｜ 生成 20s × (1+1) = 40s
    （改之前：判定 60s、生成 60s × (1+3) = 180s+）
    """
    s = get_settings()
    assert s.llm_timeout_judge == 8.0
    assert s.llm_max_retries_judge == 0, "延迟敏感路径不该有重试"
    assert s.llm_timeout_generate == 20.0
    assert s.llm_max_retries_generate == 1
    assert s.llm_max_tokens == 2048
    # 两条路径**有意不同档**：延迟敏感的判定/精排必须严格快于生成
    assert s.llm_timeout_judge < s.llm_timeout_generate


def test_generate_path_uses_its_own_timeout_and_retries(monkeypatch):
    """生成路径的取值来自配置，**不是**硬编码。

    桩值刻意取成 11.0 / 4（与生产默认 20.0 / 1 都不一样）—— 这样若有人把
    `_build_model` 改回硬编码，模型上读到的会是 20/1 而不是 11/4，测试立刻红。
    生产默认值本身由 `test_default_budget_is_bounded` 钉住。
    """
    stub = _StubSettings()
    monkeypatch.setattr(graph_builder, "get_settings", lambda: stub)
    model = graph_builder._build_model()
    assert model.request_timeout == 11.0
    assert model.max_retries == 4


def test_thinking_enabled_warns_once(monkeypatch):
    """⑩：主动开思考必须**响**一次 —— 那是 content 被吃空的原始触发条件。"""
    events: list[str] = []
    monkeypatch.setattr(litellm_client, "_thinking_warned", False)
    monkeypatch.setattr(litellm_client, "audit", lambda e, **kw: events.append(e))
    monkeypatch.setattr(litellm_client, "get_settings",
                        lambda: _stub_with(llm_thinking_enabled=True))
    litellm_client.warn_if_thinking_enabled()
    litellm_client.warn_if_thinking_enabled()   # 幂等：不该重复响
    assert events == ["thinking_enabled_warning"]


def test_thinking_disabled_is_silent(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(litellm_client, "_thinking_warned", False)
    monkeypatch.setattr(litellm_client, "audit", lambda e, **kw: events.append(e))
    monkeypatch.setattr(litellm_client, "get_settings",
                        lambda: _stub_with(llm_thinking_enabled=False))
    litellm_client.warn_if_thinking_enabled()
    assert events == []


def _stub_with(**overrides) -> _StubSettings:
    stub = _StubSettings()
    for k, v in overrides.items():
        setattr(stub, k, v)
    return stub
