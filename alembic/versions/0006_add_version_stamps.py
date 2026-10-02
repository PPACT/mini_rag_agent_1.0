"""`2.0-30`：给 `documents` / `chunks` 补**版本戳 + 内容哈希** —— 增量重跑的地基。

| 列 | 表 | 来源 | 用途 |
|---|---|---|---|
| `source_hash` | `documents` | 源文件**字节**的 sha256 | 这份文件换过没有 |
| `parser_version` | `documents` | `src/provenance.py` 常量 | 解析层口径 |
| `clean_rules_version` | `documents` | 同上（现在是 `none`） | 清洗层口径 |
| `chunker_version` | `documents` | 同上 | 切块层口径 |
| `embedding_model` | `documents` | `provenance.embedding_id()` | **模型名＋维度** |
| `content_status` | `documents` | `active` / `stale` / `deprecated` | 内容新鲜度 |
| `content_hash` | `chunks` | 块文本 `content` 的 sha256 | **只重嵌变了的块** |
| `table_id` | `chunks` | 预留（`2.0-1` 表格） | 同一张表的多个块能归组 |
| `image_path` | `chunks` | 预留（`2.0-21` 图片） | 图片块的落盘位置 |

⚠️ **新列叫 `content_status`，不叫 `status`** ——
`documents.status` 在迁移 `0001` 就已经存在，装的是**处理状态**
（`pending`/`processing`/`completed`/`failed`/`deleted`，还带 CHECK 约束）。
那是**另一个概念**：复用它会把"**跑完了没**"和"**还算不算数**"搅成一个字段，
将来任一方扩展都会踩到另一方。**`0001` 的 `status` 一个字不动。**

⚠️ **⛔ 不动 `chunks.is_deprecated`**（文档侧点名）——
检索热路径恒带 `c.is_deprecated = false` 硬过滤（`pg_vector.py`）。
本轮 `is_deprecated`（**管检索过滤**）与 `content_status`（**只标记、不参与过滤**）**并存**，
稳定之后再谈收敛。

⚠️ **`clean_rules_version` 落的是 `none`**（`2.0-5` 未做）——
**不拿空串或 `None` 假装"清洗过了"**：`none` 是"确实没有清洗层"这个**事实**。

⚠️ **本轮不加索引** —— 还没有查"哪些文档 stale"的代码路径，
先加索引只是白白增加写放大。等真有那条查询再加。

⚠️ **老行不回填**：新增列一律可空（`content_status` 除外，它有默认值）。
回填假数据会让"这行是旧管道的"这个**事实**消失 ——
`source_hash` 为 `NULL` **本身**就是"还不认识这份文档"的正确表达。

Revision ID: 0006
Revises: 0005
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_DOC_COLS = (
    "source_hash", "parser_version", "clean_rules_version",
    "chunker_version", "embedding_model",
)
_CHUNK_COLS = ("content_hash", "table_id", "image_path")


def upgrade() -> None:
    for col in _DOC_COLS:
        op.execute(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS {col} TEXT")
    op.execute(
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_status "
        "TEXT NOT NULL DEFAULT 'active'"
    )
    # 取值钉住：写错了当场报错，而不是静默存一个拼错的字符串进去。
    # ⚠️ `ADD CONSTRAINT` 没有 `IF NOT EXISTS` → 用 DO 块吞掉"已存在"（重跑安全）。
    op.execute(
        """
        DO $$ BEGIN
            ALTER TABLE documents ADD CONSTRAINT ck_documents_content_status
                CHECK (content_status IN ('active', 'stale', 'deprecated'));
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        """
    )
    for col in _CHUNK_COLS:
        op.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_content_status")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS content_status")
    for col in reversed(_DOC_COLS):
        op.execute(f"ALTER TABLE documents DROP COLUMN IF EXISTS {col}")
    for col in reversed(_CHUNK_COLS):
        op.execute(f"ALTER TABLE chunks DROP COLUMN IF EXISTS {col}")
