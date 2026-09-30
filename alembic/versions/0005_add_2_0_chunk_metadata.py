"""给 chunks 补 2.0 的解析元数据：page / raw_table / table_complex。

**为什么三个一起加**：它们同源（解析层产出）、同类（chunk 级元数据）、同批上线。
拆成三个迁移只会多两次"两库各跑一次 + 校验一致性"的仪式
（见 `scripts/migrate_all.py` 里为什么那两步缺一不可）。

| 列 | 来源 | 用途 |
|---|---|---|
| `page` | PDF 逐页解析（1-based） | 引用可答"第几页"（**2.0-4**） |
| `raw_table` | 表格块的**原表 Markdown** | **只进词法、不进向量**（**2.0-1** 口径，交流区 §1.24 Q5） |
| `table_complex` | 表格复杂度判定 | 复杂表的**原表也进向量**；评测能分开看 |

⚠️ **`page` 可空，且不拿 0 假装"有页码"** ——
Word / Excel 没有稳定的页码概念（分页由渲染决定），编一个页码比没有更糟：
引用溯源会显示一个**看起来可信但其实是编造**的页码。

⚠️ **`raw_table` 只进词法**：`content_tsv = tokenize(content + raw_table)`，
而 `embedding` 仍只算 `content`。理由见 `src/document_parser/tables.py` 模块 docstring ——
裸表的向量表达力弱（语义依赖行列头）、且会挤占向量 top-k 席位。

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page INTEGER")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS raw_table TEXT")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_complex BOOLEAN NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS table_complex")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS raw_table")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS page")
