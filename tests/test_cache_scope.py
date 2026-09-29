"""D9-⑦：缓存**键**与缓存**失效范围**必须同构（都按 kb 分）。

原来键里含 kb、失效却是 `rag:*` 全局 → **在压测库上传一次文档，真实库的问答缓存被全清**。
不是正确性问题（只是白重算），但两个库互相干扰，且压测流量会持续冲刷真实库的缓存。
"""
from __future__ import annotations

import asyncio
import fnmatch
import inspect

import pytest

from src.api import chat_api
from src.cache import redis_client
from src.db.kb import KB_REAL, KB_STRESS, UnknownKBError


class _FakeRedis:
    """只记录被扫的模式与被删的键，不连真 Redis。"""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self.pattern: str | None = None
        self.deleted: tuple | None = None

    async def scan_iter(self, match: str):
        self.pattern = match
        for k in self._keys:
            yield k

    async def delete(self, *keys):
        self.deleted = keys


def test_cache_key_matches_invalidation_pattern():
    """**这条是 ⑦ 的核心**：键的前缀必须落在该库的失效模式里。

    键与失效分别由两处代码生成，最危险的失败是"两边都改了但改得不一样" ——
    于是失效静默匹配不到任何键（缓存永远不过期，且不报错）。
    """
    for kb in (KB_REAL, KB_STRESS):
        key = chat_api._cache_key("年假几天", "IT", 3, 5, kb)
        assert fnmatch.fnmatch(key, f"rag:{kb}:*"), f"{key} 不在 rag:{kb}:* 的失效范围内"


def test_cache_key_differs_per_kb():
    a = chat_api._cache_key("q", "IT", 3, 5, KB_REAL)
    b = chat_api._cache_key("q", "IT", 3, 5, KB_STRESS)
    assert a != b, "键里不带 kb 会让两个库共用缓存（静默串库）"


def test_invalidate_only_touches_its_own_kb(monkeypatch):
    fake = _FakeRedis(["rag:stress:a", "rag:stress:b"])
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    asyncio.run(redis_client.invalidate_cache(KB_STRESS))
    assert fake.pattern == f"rag:{KB_STRESS}:*", "失效模式必须限定在本库"
    assert fake.deleted == ("rag:stress:a", "rag:stress:b")


def test_kb_is_required(monkeypatch):
    """`kb` 必填：给默认值就等于退回"静默全局失效"。"""
    sig = inspect.signature(redis_client.invalidate_cache)
    assert sig.parameters["kb"].default is inspect.Parameter.empty


def test_invalidate_rejects_unknown_kb(monkeypatch):
    """拼错的 kb 会静默匹配不到任何键 → 必须直接报错，而不是"清了个寂寞"。"""

    def _boom():
        raise AssertionError("kb 非法时不该去连 Redis")

    monkeypatch.setattr(redis_client, "get_redis", _boom)
    with pytest.raises(UnknownKBError):
        asyncio.run(redis_client.invalidate_cache("real_typo"))
