"""PgVectorStore 的 AccessFilter→SQL 翻译单测（硬过滤默认路径）。"""
from src.vector_store.base import AccessFilter
from src.vector_store.pg_vector import PgVectorStore


def test_build_search_sql_no_filter():
    store = PgVectorStore()
    sql, params = store._build_search_sql("[0.1,0.2]", AccessFilter(), 5)
    assert "d.is_deleted = false" in sql
    assert "c.department = ANY" not in sql
    assert "c.secret_level <=" not in sql
    assert "LIMIT $2" in sql
    assert params == ["[0.1,0.2]", 5]


def test_build_search_sql_with_filters():
    store = PgVectorStore()
    sql, params = store._build_search_sql(
        "[0.1,0.2]", AccessFilter(departments=["IT"], secret_level_le=3), 5
    )
    # 硬过滤：secret 占 $2（先于部门），部门占 $3
    assert "c.secret_level <= $2" in sql
    assert "c.department = ANY($3)" in sql
    assert "LIMIT $4" in sql
    assert params == ["[0.1,0.2]", 3, ["IT"], 5]


def test_build_search_sql_department_only():
    store = PgVectorStore()
    sql, params = store._build_search_sql("[0.1,0.2]", AccessFilter(departments=["IT"]), 5)
    assert "c.department = ANY($2)" in sql
    assert "c.secret_level <=" not in sql
    assert "LIMIT $3" in sql
    assert params == ["[0.1,0.2]", ["IT"], 5]
