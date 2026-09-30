"""全仓语法普查 —— 一道本该早就有的闸。

**为什么加它**（2026-09-30）：`eval/verify_hnsw_plan.py` 里有一个语法错误
（中文串里混了**半角双引号**，把字符串提前截断了），**在公开仓库里活了 2 天**：

- `pytest` 抓不到 —— 没有测试 import 它（它是给人手动跑的脚本）；
- 我当天的 `py_compile` 也没抓到 —— 那是在**最后一次编辑之前**跑的，
  而错误恰恰是最后一次编辑引入的（**验的是中间态，不是最终态**）；
- 之后又经历了两次推送与一次历史重写，一路带到了 2.0。

⚠️ **教训不是"下次小心点"，而是"缺一道不依赖人记得的闸"**：
只要 `pytest` 跑一次，本文件就会把这类错误拦下来。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "eval", "scripts", "tests", "alembic")


def _all_python_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.is_dir():
            files += [p for p in base.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files)


def test_repo_python_files_are_discoverable():
    """先确认普查范围不为空 —— 否则"0 个错误"毫无意义。"""
    files = _all_python_files()
    assert len(files) > 50, f"只扫到 {len(files)} 个文件，扫描范围可能写错了"
    assert any(p.name == "main.py" for p in files), "应包含 src/main.py"


def test_all_python_files_compile():
    """全仓每个 `.py` 都必须能被解析。

    这是**最低限度**的检查（只解析、不导入、不执行），因此**不需要 DB / 网络 / 密钥**，
    可以在任何环境跑 —— 也正是它该待在测试里的理由。
    """
    failures: list[str] = []
    for p in _all_python_files():
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            failures.append(f"{p.relative_to(ROOT)}:{e.lineno}  {e.msg}")
    assert not failures, "有文件语法不合法：\n  " + "\n  ".join(failures)


def test_sweep_detects_a_known_broken_snippet():
    """反向验证：**这套检查真的能发现问题**，不是永远绿。

    做法：拿一个确定非法的片段喂给同一套判据，必须失败。
    （本项目反复强调"验收脚本自身也要被验收"——这里用最小成本做到。
    用 `pytest.raises(SyntaxError)` 显式接受异常，而不是 `pytest.warns`。）
    """
    with pytest.raises(SyntaxError):
        ast.parse('print("外面"里面"外面")\n')
