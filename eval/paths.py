"""评测 / 体检脚本共用的**路径单一来源**。

⭐ **语料目录只在这里定义一次** —— 别处一律：

    from eval.paths import CORPUS_DIR

⚠️ 为什么单列一个模块（`dev-agent.md`：**同一事实只留一个来源**）：
这条路径原先**硬编码在 12 个文件里**，用户 2026-10-03 把语料移进 `docs/local/` 后
**12 处同时断链**。两处各写一份**必然漂移**，而且是**静默**漂移。

⚠️ 语料**不入库**（协议 §12.5：绝不外传）：它现在落在 `docs/local/` 下 ——
`docs/local/` 是一条**目录规则**（fail-closed），所以语料永远不会被提交。
"""
from __future__ import annotations

from pathlib import Path

# 项目根（本文件在 <root>/eval/ 下）
ROOT = Path(__file__).resolve().parents[1]

# 真实语料（用户维护，开发**只读**）—— 不入库，见模块 docstring。
CORPUS_DIR = ROOT / "docs" / "local" / "corpus"

__all__ = ["ROOT", "CORPUS_DIR"]
