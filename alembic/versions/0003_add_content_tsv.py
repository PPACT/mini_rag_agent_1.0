"""给 chunks 加全文检索列与索引（混合检索的词法侧）。

Revision ID: 0003
Revises: 0002

注意：新增列对历史数据为空，需重新灌库或回填
（`UPDATE chunks SET content_tsv = to_tsvector('simple', content)` 只对英文有效，
中文必须用 Python 侧 jieba 分词后回填）。
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
