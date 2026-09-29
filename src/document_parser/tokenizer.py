"""中文分词：为全文检索（tsvector）提供空格分隔的词序列。

为什么需要：PostgreSQL 自带的分词器不切中文（会把一整段中文当成一个 token），
所以必须在 Python 侧用 jieba 切好，再交给 `to_tsvector('simple', ...)` 建索引。
"""
from __future__ import annotations

import re

import jieba

# 只保留"有意义的"token：字母、数字、汉字（丢弃纯标点/空白）
_PUNCT_ONLY = re.compile(r"^[^\w一-鿿]+$", re.UNICODE)


def tokenize(text: str) -> str:
    """切成空格分隔的词序列，供 `to_tsvector('simple', ...)` / `to_tsquery` 使用。"""
    if not text:
        return ""
    tokens = []
    for word in jieba.cut(text):
        word = word.strip()
        if word and not _PUNCT_ONLY.match(word):
            tokens.append(word.lower())
    return " ".join(tokens)
