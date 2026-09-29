"""给 documents 加 original_name（用户上传时的原始文件名）。

**为什么拆两列**（D9-③）：

上传路径把 `documents.filename` 写成了 UUID 落盘名（`upload_api.py`），
而 worker 又把它当 `source_file` 灌进 `chunks` → 两个后果：

1. **引用溯源对外显示 `3f2a9c….docx`**，而不是"员工手册.docx"；
2. `retriever.format_context` 把 `文件:{source_file}` **写进 LLM 上下文**。

而 `eval/ingest.py` 那条入库路径存的却是真文件名 → **两条路径语义不一致**。

拆列后语义固定：

    filename      落盘名（UUID），仅供服务端定位文件，**不对外**
    original_name 原始文件名，**对外溯源 / LLM 上下文用**

⚠️ 存量行 `original_name` 为 NULL（**不拿 UUID 回填冒充原始名**——那是假数据）；
读取侧一律写 `original_name or filename` 回落。

Revision ID: 0004
Revises: 0003
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS original_name TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS original_name")
