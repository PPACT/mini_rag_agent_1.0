"""PgVectorStore 的 AccessFilter→SQL 翻译单测（硬过滤默认路径）。"""
from src.db.kb import KB_REAL
from src.vector_store.base import AccessFilter
from src.vector_store.pg_vector import PgVectorStore


def test_build_search_sql_no_filter():
    store = PgVectorStore(KB_REAL)
    sql, params = store._build_search_sql("[0.1,0.2]", AccessFilter(), 5)
    assert "d.is_deleted = false" in sql
    assert "c.department = ANY" not in sql
    assert "c.secret_level <=" not in sql
    assert "LIMIT $2" in sql
    assert params == ["[0.1,0.2]", 5]


def test_build_search_sql_with_filters():
    store = PgVectorStore(KB_REAL)
    sql, params = store._build_search_sql(
        "[0.1,0.2]", AccessFilter(departments=["IT"], secret_level_le=3), 5
    )
    # 硬过滤：secret 占 $2（先于部门），部门占 $3
    assert "c.secret_level <= $2" in sql
    assert "c.department = ANY($3)" in sql
    assert "LIMIT $4" in sql
    assert params == ["[0.1,0.2]", 3, ["IT"], 5]


def test_build_search_sql_department_only():
    store = PgVectorStore(KB_REAL)
    sql, params = store._build_search_sql("[0.1,0.2]", AccessFilter(departments=["IT"]), 5)
    assert "c.department = ANY($2)" in sql
    assert "c.secret_level <=" not in sql
    assert "LIMIT $3" in sql
    assert params == ["[0.1,0.2]", ["IT"], 5]


def test_no_soft_filter_orders_by_raw_distance():
    """D9-⑪ 回归：无软过滤时必须 `ORDER BY 原生 <=> 运算符`，否则 HNSW 索引失效。

    实测（`eval/verify_hnsw_plan.py`，2026-09-26）：`ORDER BY score DESC`（表达式形态，
    哪怕只是给距离乘 1.0）→ **Seq Scan + Sort**；换成原生运算符 → **Index Scan**。
    1231 切片下两种写法都只有几毫秒，看不出差别，**数据一涨就是数量级差异**。

    ⚠️ 这条测试是**字符串级**的防复发闸门：真计划形态的实测在 `eval/verify_hnsw_plan.py`。
    """
    sql, params = PgVectorStore(KB_REAL)._build_search_sql(
        "[0.1,0.2]", AccessFilter(departments=["IT"]), 5
    )
    assert "ORDER BY c.embedding <=> $1::vector" in sql
    assert "ORDER BY score DESC" not in sql, (
        "ORDER BY <表达式> 会让 pgvector 用不上 HNSW 索引（D9-⑪ 已实测为 Seq Scan）"
    )
    # 数值语义与参数编号都不能变：score 仍是 1 - 距离
    assert "1 - (c.embedding <=> $1::vector) AS score" in sql
    assert params == ["[0.1,0.2]", ["IT"], 5]


def test_soft_filter_keeps_weighted_order():
    """软过滤开启时确实在降权 → 只能按表达式排序（索引用不上是该形态的固有代价）。"""
    sql, params = PgVectorStore(KB_REAL)._build_search_sql(
        "[0.1,0.2]", AccessFilter(departments=["IT"]), 5, soft_scope=True
    )
    assert "ORDER BY score DESC" in sql
    assert "ORDER BY c.embedding <=> $1::vector" not in sql
    assert "CASE WHEN c.department = ANY($2)" in sql
    assert params == ["[0.1,0.2]", ["IT"], 0.85, 5]
