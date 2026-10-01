"""L0 切分体检报告（`2.0-36`）—— **零成本、纯 CPU**，不需要嵌入 / 标注 / LLM。

## 为什么要有它（`2.0-39` 的纪律）

`chunk_size` / `overlap` 是**全库最贵的旋钮** —— 一改就**全量重嵌**。
所以纪律是：**改之前必须附 L0 报告 + L1 分层数字**。
没有它，"我想把 chunk_size 从 512 调到 800" 就是**闭着眼睛烧钱**，且结论不可归因。

## 它回答什么

| 指标 | 用来发现 |
|---|---|
| 块长分布 p50/p90/p99/max | 是不是被某个巨大块拉偏了 |
| 超长块 / 极短块 / 空块占比 | 切分参数是否合适 |
| 表头是否与值同句 | 表格自然语言版的**质量**（串行 = 语义错乱） |
| 数值能否在自然语言版找到 | 表格保真（2.0-1 的验收闸） |
| 代码块是否被截断 | 代码整块策略有没有生效 |
| **每份文档实际生效的解析器** | 路由 vs 实际（协议 P-2） |

用法：
    python eval/l0_report.py                        # 默认扫 docs/corpus
    python eval/l0_report.py --dir eval/corpus      # 换目录
    python eval/l0_report.py --no-detail            # 不打印逐文件表
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.document_parser.chunking import split_blocks  # noqa: E402
from src.document_parser.routing import DocumentParseError, route_file  # noqa: E402
from src.document_parser.routing import parse_document  # noqa: E402
from src.document_parser.tables import missing_numbers  # noqa: E402

# 判据阈值（改这里就是改口径，所以写在明面上）
LONG_RATIO = 2.0        # 超过 chunk_size×2 算"超长块"
TINY_CHARS = 50         # 低于 50 字符算"极短块"
OUT_DEFAULT = ROOT / "logs" / "l0_report.json"


def pct_nearest_rank(vals: list[int], q: float) -> int:
    """**nearest-rank** 分位（`2.0-51①`）。

    ⚠️ **别改回 `statistics.quantiles`** —— 它的默认 `exclusive` 方法在**小样本上会外推**，
    实测给过 `p99 = 2932 > max = 2259`（分位比最大值还大，一眼可判），
    以及在 `[192, 2259]` 这种两点样本上把 p50 报成**两点的中点** 1225。
    → **分位数必须落在样本区间内**，否则它就不是这批数据的摘要。
    原始值随 JSON 明细一起输出（`files_detail[].lens_asc`），让数可复核。
    """
    v = sorted(vals)
    if not v:
        return 0
    return v[min(len(v) - 1, max(0, math.ceil(q / 100 * len(v)) - 1))]


def _measure(path: Path, chunk_size: int, overlap: int) -> dict:
    """单个文件的体检项。失败时返回带 `error` 的记录（**不中断整批**）。"""
    rec: dict = {"file": path.name}
    try:
        plan = route_file(str(path))
        doc = parse_document(str(path))
        chunks = split_blocks(doc, chunk_size, overlap)
    except DocumentParseError as e:
        rec.update(error=e.reason, kind="", parser="")
        return rec
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {e}", kind="", parser="")
        return rec

    lens = [len(c.text) for c in chunks]
    tables = doc.tables()
    # ⚠️ 「表格**块**」（解析层）与「表格 **chunk**」（切块层）在这里是 **1:1** ——
    # `_segments` 让每个原子块自成一段，合并不改变数量。
    # 但 **caption 只有 chunk 级才看得到**（`2.0-51②`）→ 计数统一走这一份。
    table_chunks = [c for c in chunks if c.kind == "table"]
    # 表头与值同句 + 数值保真
    table_rows = table_ok = 0
    missing = 0
    for t in tables:
        table_rows += 1
        # `render_nl` 产出 `列=值；列=值` —— 每行该有"列数"个 `=`；缺了就说明串行/丢列
        if t.text.count("=") >= max(1, t.raw_table.count("\n") - 1):
            table_ok += 1
        missing += len(missing_numbers(t.raw_table, t.text))

    rec.update(
        kind=plan.kind, parser=doc.parser,               # **实际生效的解析器**（P-2）
        warnings=list(doc.warnings),
        blocks=doc.kind_counts(),
        n_chunks=len(chunks),
        lens_p50=pct_nearest_rank(lens, 50), lens_p90=pct_nearest_rank(lens, 90),
        lens_p99=pct_nearest_rank(lens, 99), lens_max=max(lens) if lens else 0,
        n_long=sum(1 for x in lens if x > chunk_size * LONG_RATIO),
        n_tiny=sum(1 for x in lens if x < TINY_CHARS),
        n_empty=sum(1 for x in lens if x == 0),
        n_table=len(table_chunks),
        # ⚠️ **必须在「切块后」数**（`2.0-51②`）—— docx 的 caption 是
        # `chunking._merge_heading_captions` 在**切块层**贴上去的；
        # 读**块级**（`doc.tables()`）**看不见它** → 该指标会**结构性地测不到这个机制**。
        # 实测：块级恒报 20，切块后才是 22 → 28。**两个数不是一回事，别混**：
        #   · `n_cap_yes` = 拿到了 caption（含"紧邻 heading 被合并进来"的那些）
        #   · `n_cap_no`  = 没有 caption —— **多数是"本来就没有紧邻 heading"**，
        #     不是"漏判"。真正的漏判（判据该合而未合）在本判据下**恒为 0**（确定性事实判定）。
        n_cap_yes=sum(1 for c in table_chunks if c.text.startswith("表「")),
        n_cap_no=sum(1 for c in table_chunks if not c.text.startswith("表「")),
        # 原始块长（升序）—— 让分位数**可复核**，不必再拿它当黑箱（`2.0-51①`）
        lens_asc=sorted(lens),
        table_ok=table_ok,
        table_missing_numbers=missing,
        code_len_max=max([len(b.text) for b in doc.blocks if b.kind == "code"], default=0),
        n_code=sum(1 for b in doc.blocks if b.kind == "code"),
    )
    return rec


def _fmt_row(r: dict) -> str:
    if r.get("error"):
        return f"{r['file'][:30]:<32} {'—':<8}{'❌ ' + r['error'][:40]}"
    return (f"{r['file'][:30]:<32} {r['kind']:<8}{r['parser']:<12}"
            f"{r['n_chunks']:>4} {r['lens_p50']:>6}{r['lens_p90']:>6}{r['lens_max']:>7} "
            f"{r['n_long']:>4} {r['n_tiny']:>4} {r['n_table']:>4} "
            f"{'✅' if r['table_missing_numbers'] == 0 else '❌'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "docs" / "corpus"))
    ap.add_argument("--json", default=str(OUT_DEFAULT))
    ap.add_argument("--no-detail", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"❌ 目录不存在：{src_dir}")
        return 1
    files = sorted(p for p in src_dir.iterdir() if p.is_file())

    recs = [_measure(p, s.chunk_size, s.chunk_overlap) for p in files]
    ok = [r for r in recs if not r.get("error")]
    failed = [r for r in recs if r.get("error")]

    print(f"L0 切分体检 ｜ 目录 {src_dir} ｜ {len(files)} 个文件"
          f"（成功 {len(ok)} / 失败 {len(failed)}）")
    print(f"参数：chunk_size={s.chunk_size}  overlap={s.chunk_overlap}  "
          f"（超长阈值 >{int(s.chunk_size * LONG_RATIO)} 字符；极短 <{TINY_CHARS} 字符）\n")

    if not args.no_detail:
        print(f"{'文件':<32} {'路由':<8}{'解析器':<12}{'块数':>4} "
              f"{'p50':>6}{'p90':>6}{'max':>7} {'超长':>4} {'极短':>4} {'表':>4} 保真")
        print("-" * 108)
        for r in recs:
            print(_fmt_row(r))
        print("-" * 108)

    # ---- 汇总 ----
    n_chunks = sum(r["n_chunks"] for r in ok)
    n_long = sum(r["n_long"] for r in ok)
    n_tiny = sum(r["n_tiny"] for r in ok)
    n_empty = sum(r["n_empty"] for r in ok)
    n_table = sum(r["n_table"] for r in ok)
    n_cap_yes = sum(r["n_cap_yes"] for r in ok)
    n_cap_no = sum(r["n_cap_no"] for r in ok)
    miss = sum(r["table_missing_numbers"] for r in ok)
    n_code = sum(r["n_code"] for r in ok)
    code_max = max([r["code_len_max"] for r in ok], default=0)
    n_warn = sum(len(r["warnings"]) for r in ok)

    print(f"\n【汇总】共 {n_chunks} 个块")
    print(f"  超长块（>{int(s.chunk_size * LONG_RATIO)} 字符）：{n_long}  "
          f"({n_long / n_chunks * 100:.1f}%)" if n_chunks else "  超长块：—")
    print(f"  极短块（<{TINY_CHARS} 字符）：{n_tiny}"
          f"{f'  ({n_tiny / n_chunks * 100:.1f}%)' if n_chunks else ''}"
          f" ｜ 空块：{n_empty}")
    print(f"  表格块：{n_table}｜**数值保真失败 {miss}**（应为 0）")
    print(f"    · 含 caption（`表「X」：`，`2.0-46`，**切块后计数**）：{n_cap_yes}"
          f" ｜ 无 caption：{n_cap_no}（**多数是本来就没有紧邻 heading**，不是漏判；"
          f"判据该合而未合 = 0，确定性判定）")
    print(f"  代码块：{n_code} 个，最长 {code_max} 字符"
          f"{'  ← ⚠️ 整块未切（2.0-24 后半未做，属预期）' if code_max > s.chunk_size * 2 else ''}")
    print(f"  解析警告：{n_warn} 条")
    print(f"  失败文件：{len(failed)}"
          + (f" → {', '.join(r['file'] for r in failed[:5])}" if failed else ""))

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"params": {"chunk_size": s.chunk_size, "overlap": s.chunk_overlap,
                    "long_ratio": LONG_RATIO, "tiny_chars": TINY_CHARS},
         "summary": {"files": len(files), "ok": len(ok), "failed": len(failed),
                     "chunks": n_chunks, "long": n_long, "tiny": n_tiny, "empty": n_empty,
                     "tables": n_table, "table_missing_numbers": miss,
                     "tables_with_caption": n_cap_yes, "tables_without_caption": n_cap_no,
                     "code_blocks": n_code, "code_max": code_max, "warnings": n_warn},
         "files_detail": recs},
        ensure_ascii=False, indent=1), encoding="utf-8")
    # ⚠️ 输出路径可能在项目外（测试用临时目录、或用户指定别处）→
    # `relative_to` 会抛 ValueError，而且**是在干完所有活之后**才抛（测试抓到的）。
    try:
        shown: object = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\n逐文件明细 → {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
