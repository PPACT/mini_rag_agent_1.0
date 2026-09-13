"""分词与词法检索的单元测试。"""
from src.document_parser.tokenizer import tokenize
from src.vector_store.base import AccessFilter
from src.vector_store.pg_vector import PgVectorStore


# ---- 分词 ----

def test_tokenize_chinese():
    tokens = tokenize("密码每90天强制更换一次").split()
    assert "密码" in tokens
    assert "90" in tokens
    assert "更换" in tokens


def test_tokenize_drops_punctuation():
    tokens = tokenize("密码，长度不少于 12 位。").split()
    assert "，" not in tokens
    assert "。" not in tokens
    assert "12" in tokens


def test_tokenize_keeps_acronyms_lowercased():
    tokens = tokenize("灾备切换的 RTO 不能超过").split()
    assert "rto" in tokens  # 统一小写，与 to_tsvector('simple') 行为一致


def test_tokenize_empty():
    assert tokenize("") == ""
    assert tokenize("   ，。！   ") == ""


# ---- 词法检索 SQL ----

def test_build_lexical_sql_no_filter():
    sql, params = PgVectorStore._build_lexical_sql("密码 | 更换", AccessFilter(), 5)
    assert "c.content_tsv @@ to_tsquery('simple', $1)" in sql
    assert "ORDER BY score DESC" in sql
    assert "LIMIT $2" in sql
    assert params == ["密码 | 更换", 5]


def test_build_lexical_sql_with_filters():
    sql, params = PgVectorStore._build_lexical_sql(
        "密码", AccessFilter(departments=["IT"], secret_level_le=3), 5
    )
    assert "c.department = ANY($2)" in sql
    assert "c.secret_level <= $3" in sql
    assert "LIMIT $4" in sql
    assert params == ["密码", ["IT"], 3, 5]


# ---- 关键：两侧权限过滤必须语义一致（抽象边界要求）----

def test_permission_conds_identical_between_vector_and_lexical():
    """向量检索与词法检索必须用**同一套**权限条件，否则会出现安全漏洞。"""
    filters = AccessFilter(departments=["IT", "HR"], secret_level_le=2)
    vec_sql, _ = PgVectorStore._build_search_sql("[0.1,0.2]", filters, 5)
    lex_sql, _ = PgVectorStore._build_lexical_sql("密码", filters, 5)

    for cond in ("d.is_deleted = false", "c.is_deprecated = false",
                 "c.department = ANY($2)", "c.secret_level <= $3"):
        assert cond in vec_sql, f"向量侧缺少: {cond}"
        assert cond in lex_sql, f"词法侧缺少: {cond}"


def test_permission_conds_identical_no_filter():
    """无过滤时两侧都不应出现权限条件，且排除条件一致。"""
    vec_sql, _ = PgVectorStore._build_search_sql("[0.1]", AccessFilter(), 5)
    lex_sql, _ = PgVectorStore._build_lexical_sql("x", AccessFilter(), 5)
    for cond in ("d.is_deleted = false", "c.is_deprecated = false"):
        assert cond in vec_sql
        assert cond in lex_sql
    assert "c.department = ANY" not in vec_sql
    assert "c.department = ANY" not in lex_sql
