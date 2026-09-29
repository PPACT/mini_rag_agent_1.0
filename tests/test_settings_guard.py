"""D9-⑥ 的附加项：把「选了未实现的引擎」挡在**启动期**。

`MilvusStore` 是占位桩（所有方法 raise NotImplementedError），而
`get_vector_store` 只看 `VECTOR_STORE` 就把它返回 —— 选了 milvus 之后
启动、建连、创建文档全部正常，**直到第一次检索才炸**，是典型的静默故障。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import Settings


def test_milvus_backend_is_rejected_at_config_load():
    with pytest.raises(ValidationError) as ei:
        Settings(vector_store="milvus")
    assert "尚未接入" in str(ei.value), "报错要说清原因，别让人去翻源码"


def test_pgvector_backend_is_accepted():
    assert Settings(vector_store="pgvector").vector_store == "pgvector"


def test_guard_is_case_insensitive():
    with pytest.raises(ValidationError):
        Settings(vector_store="Milvus")
