"""给 chunks 补元数据：start_offset / end_offset / title / is_deprecated。

Revision ID: 0002
Revises: 0001
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_offset INTEGER")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_offset INTEGER")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS title TEXT")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_deprecated BOOLEAN NOT NULL DEFAULT false")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_is_deprecated ON chunks (is_deprecated)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_is_deprecated")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS is_deprecated")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS title")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS end_offset")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS start_offset")
