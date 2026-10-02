"""`2.0-33` 元数据更新通道的**闸**。

⭐ 这一套的全部价值就是**白名单**：让"改元数据"这条路**不可能**碰到 `content` / `embedding`。
⚠️ 所以**必须有一条会失败的反向用例** —— 喂一个禁止字段、它**必须报错**。
没有这条，白名单就只是一句注释（本项目反复吃过这个亏）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.update_metadata import (  # noqa: E402
    _CHUNK_FIELDS,
    _DOC_FIELDS,
    build_update,
    coerce,
    parse_set,
)


def test_parse_set_requires_key_value():
    assert parse_set(["department=HR", "secret_level=2"]) == {"department": "HR", "secret_level": "2"}
    assert parse_set([]) == {}
    with pytest.raises(ValueError, match="key=value"):
        parse_set(["department"])


# ---------------------------------------------------------------- ⭐ 反向用例


@pytest.mark.parametrize("bad", ["content", "embedding", "content_tsv", "content_hash", "raw_table"])
def test_forbidden_fields_are_rejected(bad):
    """⭐ **反向用例**：这些列**改了就作废**，必须**报错**而不是静默跳过。

    ⚠️ "静默跳过"是最坏的形态：命令看起来成功了，而调用方以为改到了。
    """
    with pytest.raises(ValueError, match="不属于元数据"):
        coerce({bad: "x"}, _CHUNK_FIELDS)


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="未知字段"):
        coerce({"deparment": "HR"}, _CHUNK_FIELDS)     # 拼错也要挡


def test_bool_and_int_are_coerced():
    assert coerce({"is_deprecated": "true"}, _CHUNK_FIELDS) == {"is_deprecated": True}
    assert coerce({"is_deprecated": "0"}, _CHUNK_FIELDS) == {"is_deprecated": False}
    assert coerce({"secret_level": "3"}, _CHUNK_FIELDS) == {"secret_level": 3}
    with pytest.raises(ValueError, match="布尔"):
        coerce({"is_deprecated": "yes"}, _CHUNK_FIELDS)


# ---------------------------------------------------------------- SQL 生成


def test_build_update_only_names_whitelisted_columns():
    """⭐ 结构保证：**SET 子句只由传进来的字段拼出** —— `embedding` 没有任何进入路径。"""
    sql, params = build_update("chunks", {"department": "HR", "secret_level": 2})
    assert sql == "UPDATE chunks SET department = $2, secret_level = $3 WHERE document_id = $1::uuid"
    assert params == ["HR", 2], "参数顺序必须与排序后的列一致"
    for forbidden in ("embedding", "content", "content_tsv"):
        assert forbidden not in sql.split("WHERE")[0], f"{forbidden} 混进了 SET 子句"


def test_build_update_refuses_empty():
    with pytest.raises(ValueError, match="没有要改的字段"):
        build_update("chunks", {})


def test_doc_fields_are_their_own_whitelist():
    """`content_status` 在 `documents` 上 —— 两张表的白名单**不许混用**。"""
    assert set(_DOC_FIELDS) == {"content_status"}
    assert "content_status" not in _CHUNK_FIELDS
    with pytest.raises(ValueError, match="未知字段"):
        coerce({"content_status": "stale"}, _CHUNK_FIELDS)
