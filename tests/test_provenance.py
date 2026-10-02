"""`2.0-30` 版本戳与内容哈希的回归闸。

⚠️ 这里**不测数据库**（迁移本身由 `scripts/migrate_all.py` 两库跑 + 校验），
测的是**那些值的语义**：它们必须是稳定的、可复现的、且**只有一个来源**。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import provenance  # noqa: E402


def test_sha256_text_is_stable_and_sensitive():
    a = provenance.sha256_text("三、成本与费用")
    assert a == provenance.sha256_text("三、成本与费用"), "同样的文本必须得到同样的哈希"
    assert len(a) == 64 and all(c in "0123456789abcdef" for c in a)
    # ⚠️ 反向用例：改一个字必须变 —— 否则"只重嵌变了的块"会漏掉真的变了的块
    assert a != provenance.sha256_text("三、成本与费用。")
    assert a != provenance.sha256_text("三、 成本与费用")


def test_sha256_file_reads_bytes(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello\x00world\xff")
    expect = hashlib.sha256(b"hello\x00world\xff").hexdigest()
    assert provenance.sha256_file(p) == expect, "必须按**字节**算（含 \\x00 / 非 UTF-8）"
    # 反向用例：内容变了哈希必须变
    p.write_bytes(b"hello\x00world\xfe")
    assert provenance.sha256_file(p) != expect


def test_embedding_id_carries_model_and_dim():
    """⚠️ 只比模型名会漏掉"同名换维度"（向量不兼容，必须重嵌）。"""
    from src.config.settings import get_settings

    s = get_settings()
    eid = provenance.embedding_id()
    assert eid == f"{s.embedding_model}:{s.embedding_dim}"
    assert ":" in eid, "维度必须进标识 —— 否则 bge-m3 换 512 维会被当成没变"


def test_document_stamps_all_present(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    st = provenance.document_stamps(p)
    assert set(st) == {"source_hash", "parser_version", "clean_rules_version",
                       "chunker_version", "embedding_model"}
    assert all(v for v in st.values()), f"不允许空值（空值会让'这行是旧管道的'这个事实消失）: {st}"
    assert st["source_hash"] == provenance.sha256_file(p)


def test_clean_rules_version_is_explicitly_none():
    """`2.0-5` 未做 —— ⚠️ 必须是显式的 `none`，**不许拿空串假装"清洗过了"**。"""
    assert provenance.CLEAN_RULES_VERSION == "none"


def test_ingest_paths_share_one_source():
    """⭐ **同一事实只留一个来源**：两条入库路径必须**引用**同一个函数，不许各写一份。

    与 `test_pct_has_a_single_source` 同款 —— 用 `is` 比**同一个函数对象**，
    复制一份"实现一样"的也会红（本项目因"两份清单必然漂移"已吃过两次）。
    """
    import eval.ingest as ing

    assert ing.sha256_text is provenance.sha256_text
    assert ing.document_stamps is provenance.document_stamps

    import src.tasks.document_task as dt

    assert dt.sha256_text is provenance.sha256_text
    assert dt.document_stamps is provenance.document_stamps
