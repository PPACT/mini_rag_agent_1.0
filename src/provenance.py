"""`2.0-30`：**管道版本戳 + 内容哈希** —— 让"改了口径只重跑受影响的部分"成为可能。

## 为什么要有它

改一次清洗规则 / 切块参数 / 嵌入模型，**旧块就不再对应当前口径**。
没有版本戳，唯一安全做法是**全量重跑**（还容易忘）；有了它，就能：
**"哪个版本生产的块还留在库里"** → 只重跑那一部分。

⚠️ 与本项目 `2.0-31` 的**约束**配套：**新增内容必须成「新的块」**，
不许并进已有 chunk 的 `content` —— 否则那个块内容一变就得重嵌。

## ⚠️ 四条版本戳的**唯一来源**就在这里

散在三个文件里各写一个常量，**必然漂移**（本项目已因"同一事实两个来源"吃过两次：
`upload_api._ALLOWED` 的两份清单 / `parse_plain` 的两处扩展名判断）。

**升版规则**（一句话）：
> **改了"会影响块文本或向量"的行为，就必须手动升对应的版本号。**
> 版本号**不是自动计算的** —— 它是个**承诺**：数字没变，就代表"这类改动没发生"。

| 常量 | 覆盖什么 | 什么时候要升 |
|---|---|---|
| `PARSER_VERSION` | 解析层：`parsers` / `table_recovery` / `routing` | 改了解析出的块（文本 / 类型 / 表格渲染） |
| `CHUNKER_VERSION` | 切块层：`chunking` / `semantic_splitter` | 改了切块边界或 `title` 归属 |
| `CLEAN_RULES_VERSION` | 清洗层（`2.0-5`，**未做**） | 清洗规则一变就升；**现在是 `none`** |

⚠️ `embedding_model` **不在这里** —— 它来自配置（`settings.embedding_model` + 维度），
配置一改就该重嵌。用 `embedding_id()` 取"**实际生效**"的那一个（协议 P-2）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# ⚠️ 升版 = 声明"块文本/向量可能变了"，会让增量重跑**重跑那一部分**。
# 别为了"顺手改个注释"升它 —— 那会让整库被判定为 stale。
PARSER_VERSION = "2.0.0"
CHUNKER_VERSION = "2.0.0"

# 清洗层 `2.0-5` 尚未实现 —— 显式写 `none`，**不拿空串假装一致**
CLEAN_RULES_VERSION = "none"

_HASH_CHUNK = 1 << 20   # 1 MiB：大文件也不整块读进内存


def sha256_file(path: str | Path) -> str:
    """源文件的 sha256（**按字节**）。

    用途：判断"这份文件是不是换过了"。⚠️ 它看的是**原始字节**，
    所以同一份内容另存一次（时间戳/元数据变了）也会得到不同的 hash —— 这是有意的：
    `source_hash` 回答的是"我拿到的还是不是同一个文件"，不是"语义是否相同"。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_HASH_CHUNK):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """`content_hash`：块文本的 sha256（UTF-8）。

    ⚠️ 只算 **`content`（自然语言版）**，**不算 `raw_table`** ——
    向量只吃 `content`（`2.0-1` 口径），所以"要不要重嵌"也只该看它。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_id() -> str:
    """**实际生效**的嵌入标识（模型名 + 维度）—— 协议 P-2：别只信配置里写了什么。

    带上维度：同一个模型名换了维度（如 `bge-m3` 截断到 512）向量**不兼容**，
    只比模型名会漏掉这种"看起来一样、其实对不上"的情况。
    """
    from src.config.settings import get_settings

    s = get_settings()
    return f"{s.embedding_model}:{s.embedding_dim}"


def document_stamps(source_path: str | Path) -> dict[str, str]:
    """一份文档入库时要落的**版本戳**（`documents` 表的那几列）。

    ⚠️ **两条入库路径（上传 / 评测灌库）都调这一个函数** ——
    别各写一份（`2.0-30` 的验收就是"两库都能查到这些列"）。
    """
    return {
        "source_hash": sha256_file(source_path),
        "parser_version": PARSER_VERSION,
        "clean_rules_version": CLEAN_RULES_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": embedding_id(),
    }
