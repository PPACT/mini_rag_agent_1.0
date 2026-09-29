"""入队侧契约测试（D9-①）。

`enqueue_process_document` 改**关键字传参**后，「位置错位」这类静默故障消失了，
但换来了另一种漂移风险：**入队方发送的键**与 **worker 形参名**可以对不上
（位置传参下，"顺序"至少是被隐式钉住的）。

本文件把两侧钉在一起——任何一边改名/加必填参数，这里立刻红。
"""
from __future__ import annotations

import asyncio
import inspect

from src.tasks import queue as queue_mod
from src.tasks.document_task import process_document


class _FakeArq:
    """只记录调用，不碰 Redis。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))


def _enqueue(monkeypatch, **call_kwargs) -> tuple[str, tuple, dict]:
    """跑一次 enqueue_process_document，返回 (job_name, args, kwargs)。"""
    fake = _FakeArq()

    async def _fake_get_arq():
        return fake

    monkeypatch.setattr(queue_mod, "get_arq", _fake_get_arq)
    asyncio.run(queue_mod.enqueue_process_document(**call_kwargs))
    assert len(fake.calls) == 1, "应当恰好入队一次"
    return fake.calls[0]


def test_enqueue_uses_keywords_only(monkeypatch):
    """D9-① 回归：不得再出现位置参数——位置错位会静默灌错库，且不报错。"""
    name, args, kwargs = _enqueue(
        monkeypatch, kb="real", document_id="doc-1", department="IT", secret_level=2
    )
    assert name == "process_document"
    assert args == (), "入队参数必须全部走关键字；位置参数会让 kb 静默错位"
    assert kwargs == {
        "kb": "real",
        "document_id": "doc-1",
        "department": "IT",
        "secret_level": 2,
    }


def test_enqueue_keys_match_worker_signature(monkeypatch):
    """入队键集合 == worker 形参名集合（任一边改名即红）。"""
    _, _, kwargs = _enqueue(
        monkeypatch, kb="stress", document_id="doc-9", department=None, secret_level=0
    )
    params = list(inspect.signature(process_document).parameters)
    assert params[0] == "ctx", "worker 首参必须是 arq 注入的 ctx"
    assert set(kwargs) == set(params[1:]), (
        f"入队键 {sorted(kwargs)} 与 worker 形参 {sorted(params[1:])} 不一致——"
        f"改了一边就要同步另一边"
    )


def test_kb_is_required():
    """kb 必填：不设默认值，防止「忘传 → 静默落到真实库」。"""
    sig = inspect.signature(queue_mod.enqueue_process_document)
    assert sig.parameters["kb"].default is inspect.Parameter.empty
