"""切块质量检测：区分「检索失败」与「切块把答案切断了」。

背景：评测里 `answer_span` 若被切块边界切断，会被记为"未命中"，
但根因在**切块**而非检索——我们需要把这两类失败分开。

方法（纯本地，不调检索/LLM）：
  1. 对每道题的期望文档，加载全文
  2. 定位 answer_span 在原文中的字符区间 [start, end)
  3. 用当前切块配置切分，检查是否存在**某一块的 [start,end) 完全包含该区间**
     - 存在 → 该题答案「完整落块」（检索可能命中）
     - 不存在 → 该题答案「被切断」（检索永远不可能命中，属于切块缺陷）

用法：python eval/run_chunk_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from src.config.settings import get_settings  # noqa: E402
from src.document_parser.loader import load_text  # noqa: E402
from src.document_parser.semantic_splitter import split_text  # noqa: E402

DATASETS = ["dataset_clear.jsonl", "dataset_exact.jsonl", "dataset_paraphrase.jsonl"]
CORPUS_DIRS = [EVAL_DIR / "corpus", EVAL_DIR / "corpus_synth"]


def _load_doc(source: str) -> str:
    for d in CORPUS_DIRS:
        p = d / source
        if p.exists():
            return load_text(str(p))
    return ""


def _check_answer(text: str, full: str, chunks) -> str:
    """返回: 'ok'(完整落块) / 'cut'(被切断) / 'missing'(片段不在原文)"""
    pos = full.find(text)
    if pos < 0:
        return "missing"
    start, end = pos, pos + len(text)
    for c in chunks:
        if c.start <= start and end <= c.end:
            return "ok"
    return "cut"


def main() -> None:
    settings = get_settings()
    total = {"ok": 0, "cut": 0, "missing": 0}
    detail: list[tuple] = []

    for ds in DATASETS:
        path = EVAL_DIR / ds
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            full = _load_doc(r["source"])
            if not full:
                total["missing"] += 1
                detail.append((ds, r["id"], r["question"][:24], "文档不存在"))
                continue
            chunks = split_text(full, settings.chunk_size, settings.chunk_overlap)
            state = _check_answer(r["answer_span"], full, chunks)
            total[state] += 1
            if state != "ok":
                detail.append((ds, r["id"], r["question"][:24], state))

    print(f"检查数据集: {', '.join(DATASETS)}（切块 {settings.chunk_size}/{settings.chunk_overlap}）\n")
    n = sum(total.values())
    print(f"{'结果':<12}{'题数':<8}{'占比':<10}")
    print("-" * 30)
    for k, label in (("ok", "完整落块（检索可达）"), ("cut", "被切断（切块缺陷）"), ("missing", "片段缺失/文档无")):
        print(f"{k:<12}{total[k]:<8}{total[k] / n if n else 0:<10.1%}")

    cuts = [d for d in detail if d[3] == "cut"]
    print(f"\n=== 被切断的题（{len(cuts)} 条，这是「检索救不回来的」切块缺陷）===")
    for ds, i, q, _ in cuts[:20]:
        print(f"  {ds} #{i} {q}")

    print(f"\n解读：'被切断' 的题无论检索多好都不可能命中 → 根因在切块，应调整切块策略或标注。")
    print(f"      '完整落块' 但检索未命中 → 才是检索层的问题。")


if __name__ == "__main__":
    main()
