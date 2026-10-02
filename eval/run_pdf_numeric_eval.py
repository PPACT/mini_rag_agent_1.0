"""`2.0-29` 评测 —— PDF 数值题的**四问**（`§1.62`）。

## 它回答什么

「PDF 抽文本时**数值与标签被拆散**」这件事，**值不值得修**。
文档侧把判读**预先定死了**（免得数字出来再吵）：

| ① R@K | ② 同块率 | ③ 残句率 | 结论 |
|---|---|---|---|
| 好 | 高 | 低 | ✅ 不用修 |
| 差 | 低 | 高 | 🔴 必修（per-document 自适应容差，**不是 MinerU**）|

## ⚠️ 我补了一问 ②b：**同行率**

**文档侧的 ② 是「同一块」—— 而 `chunk_size` 是 512，整节内容本来就都在一块里。**
实测一张住宿费标准表：数值与它上方的类别标签**确实同块**，
但**分在两行** —— ②「同块率」会是 **100% 而毫无区分度**。

→ 所以补 **②b 同行率**（数值与标签**在同一行**）——**那才是"读得通"的最小单位**。
⚠️ 四问**一起看**：② 高 + ②b 低 = **"同块但不同行"，块大掩盖了问题**。

## ⚠️ 评测集**不入库** —— 它的内容就是真实语料原文

落点：`eval/local/dataset_pdf_numeric.jsonl`（`eval/local/` 已 gitignore）。
**协议 §12.5：语料绝不外传**；且 `docs/corpus/` 本身不入库 →
**新克隆跑不了这个评测**，提交评测集换不来可复现性，只换来泄露。

每行一题（JSONL）：

    id / kind / question          题面
    source                        语料文件名
    anchor                        ⭐ 库里**带洞**的原文片段 —— 是**定位器**，不是「正确文本」
    value                         正确答案（⚠️ **故意不参与自检**，见 `selfcheck`）
    broken                        值被挪走后出现的症状串

⚠️ `anchor` 与 `broken` **必须能在库里找得到**（`selfcheck` 会拦下），否则先灌库。
📌 同理：**`2.0-7` 的 50 题也放这个目录**（它们同样带语料 anchor）。

## 用法（**跑前自动过闸** —— 记在文档里的规矩拦不住遗忘）

    python eval/run_pdf_numeric_eval.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.connection import close_pool  # noqa: E402
from src.db.kb import KB_STRESS  # noqa: E402
from src.rag.retriever import retrieve  # noqa: E402
from src.vector_store.base import Chunk  # noqa: E402

# ⚠️ 放在**已忽略**目录里 —— 题面 / anchor / value 全是真实语料原文（协议 §12.5）。
#    `eval/local/` 是一条**目录规则**（fail-closed）：以后同类的评测集丢进去就自动忽略，
#    不会因为"忘了加一行精确路径"而静默泄漏。
DATASET = Path(__file__).resolve().parent / "local" / "dataset_pdf_numeric.jsonl"
CORPUS = str(ROOT / "docs" / "corpus")
EVAL_DEPARTMENT = ["IT", "公司"]
EVAL_SECRET_LEVEL = 3
KS = (1, 3, 5)
MODE = {"use_rewrite": False, "use_rerank": False, "use_hybrid": False}   # 基线（同 2.0-47）


def load_dataset() -> list[dict]:
    # ⛔ 不做"假装支持"：缺文件就**显式报错**并说清怎么建，别静默跳过变成 0 题。
    if not DATASET.exists():
        raise SystemExit(
            f"❌ 找不到评测集：{DATASET}\n"
            "   它**不入库**（内容 = 真实语料原文，协议 §12.5）——\n"
            "   换语料时请按本文件 docstring 的字段格式自建一份放进 eval/local/。")
    return [json.loads(ln) for ln in DATASET.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def _file_chunks(kb: str) -> dict[str, list[str]]:
    """库里每份文档的 chunk 文本 —— **口径自检与判定都用这一份**（= 检索真正看到的东西）。"""
    from src.db.connection import get_pool

    pool = await get_pool(kb)
    rows = await pool.fetch(
        "SELECT COALESCE(d.original_name, d.filename) AS f, c.content AS t "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        "ORDER BY d.original_name, c.chunk_index")
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r["f"], []).append(r["t"])
    return out


def selfcheck(dataset: list[dict], by_file: dict[str, list[str]]) -> list[str]:
    """⭐ 开跑前核对：`anchor` 与 `broken` **必须**在库里找得到。

    ⚠️ **`value` 故意不核对** —— 「找不到它」**正是本题要量的东西**，
    把它列进自检就会把结论当成错误排除掉。
    """
    bad = []
    for row in dataset:
        texts = by_file.get(row["source"], [])
        for field in ("anchor", "broken"):
            if not any(row[field] in t for t in texts):
                bad.append(f"#{row['id']} {row['source']} 的 {field} 在库里找不到：{row[field]!r}")
    return bad


def rank_of(chunks: list[Chunk], source: str, anchor: str) -> int | None:
    for i, c in enumerate(chunks, start=1):
        if c.source_file == source and anchor in (c.content or ""):
            return i
    return None


def judge(chunks: list[Chunk], row: dict, rank: int | None) -> dict:
    """命中块上的三问 + 我补的同行率。"""
    if rank is None:
        return {"same_chunk": None, "same_line": None, "broken": None, "hit_text": None}
    text = chunks[rank - 1].content or ""
    lines = text.splitlines()
    return {
        "same_chunk": row["value"] in text,
        # ⭐ 我补的：数值与标签**在同一行**（"读得通"的最小单位）
        "same_line": any(row["value"] in ln and row["anchor"] in ln for ln in lines),
        "broken": row["broken"] in text,
        "hit_text": text,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default=KB_STRESS)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", default=str(ROOT / "logs" / "pdf_numeric_eval.json"))
    args = ap.parse_args()

    # ---- ⭐ 跑前先过闸（复用那道闸，不另写）----
    from eval.check_ingest_complete import check as gate

    g = await gate(args.kb, CORPUS)
    if g["missing"] or g["unexpected"] or g["empty"]:
        print("[X] **灌库不完整，先别评估**（`2.0-50`：数字会解释不通）")
        for name in g["missing"]:
            print(f"    丢了：{name}")
        return 1
    print(f"✅ 过闸：{g['known']} 个认得 / {len(g['unsupported'])} 个路由拒收 ｜ 库中 {g['in_db']} 份\n")

    dataset = load_dataset()
    by_file = await _file_chunks(args.kb)

    bad = selfcheck(dataset, by_file)
    if bad:
        print("[X] 口径自检没过 —— **不产出任何数字**（否则 ③ 恒 0 是假指标）：")
        for b in bad:
            print(f"    · {b}")
        return 1
    print(f"✅ 口径自检通过：{len(dataset)} 题的 anchor / broken 都在库里的块中找得到\n")

    print(f"【检索】基线模式 ｜ top {args.top}")
    ranks, judges = [], []
    for row in dataset:
        _, chunks = await retrieve(row["question"], EVAL_DEPARTMENT, EVAL_SECRET_LEVEL,
                                   kb=args.kb, top_k=args.top, **MODE)
        r = rank_of(chunks, row["source"], row["anchor"])
        ranks.append(r)
        judges.append(judge(chunks, row, r))

    n = len(dataset)
    print(f"  {'#':<7}{'① R':>5}{'②同块':>7}{'②b同行':>8}{'③残句':>7}   问题")
    for row, r, j in zip(dataset, ranks, judges):
        fmt = lambda v: "—" if v is None else ("✅" if v else "❌")     # noqa: E731
        print(f"  {row['id']:<7}{('—' if r is None else r):>5}{fmt(j['same_chunk']):>7}"
              f"{fmt(j['same_line']):>8}{fmt(j['broken']):>7}   {row['question'][:24]}")

    hit = [(row, r, j) for row, r, j in zip(dataset, ranks, judges) if r is not None]
    def rate(key):
        return sum(1 for _, _, j in hit if j[key]) / len(hit) if hit else 0.0

    print(f"\n【汇总 · {n} 题，命中 {len(hit)}】")
    for k in KS:
        ok = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  ① R@{k} = {ok}/{n} = {ok / n:.3f}")
    print(f"  ② 同块率（文档侧口径）= {rate('same_chunk'):.3f}")
    print(f"  ②b 同行率（**我补的**）= {rate('same_line'):.3f}   ← ⚠️ 与 ② 的差 = '同块但不同行'")
    print(f"  ③ **残句率** = {rate('broken'):.3f}   ← 句子本身已经废了")

    print("\n【命中块的原文（文档侧点名要看）】")
    for row, r, j in hit[:3]:
        print(f"  ── #{row['id']} 第 {r} 名 ｜ {row['question'][:26]}")
        print(f"     {j['hit_text'][:180]!r}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kb": args.kb, "top": args.top, "mode": "基线", "n": n,
        "ranks": ranks,
        "rates": {"same_chunk": rate("same_chunk"), "same_line": rate("same_line"),
                  "broken": rate("broken"),
                  **{f"R@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}},
        "per_question": [{**{kk: row[kk] for kk in ("id", "question", "source", "value")},
                          "rank": r, **{kk: j[kk] for kk in ("same_chunk", "same_line", "broken")},
                          "hit_text": j["hit_text"]}
                         for row, r, j in zip(dataset, ranks, judges)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        print(f"\n产物 → {out.relative_to(ROOT)}")
    except ValueError:
        print(f"\n产物 → {out}")
    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
