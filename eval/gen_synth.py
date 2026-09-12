"""生成合成语料（干扰文档），用于测试「规模对召回的影响」。

两部分：
1. **部门变体**（脚本，零成本）：把基础文档复制成各部门版本，数字微调。
   → 制造"同一制度在 N 个部门各有一版"的高密度语义竞争（真实企业最常见形态）。
2. **LLM 生成**（少量 token 成本）：生成不同主题的企业制度文档，简单但同域，增加语义竞争。

产物目录：eval/corpus_synth/
用法：
    python eval/gen_synth.py            # 两部分都生成
    python eval/gen_synth.py --skip-llm # 只生成部门变体
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.litellm_client import complete  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent / "corpus"
OUT_DIR = Path(__file__).resolve().parent / "corpus_synth"

DEPARTMENTS = [
    "IT", "人力资源", "财务", "法务", "行政", "市场", "销售", "客服", "运维", "信息安全",
    "产品", "研发", "测试", "数据", "算法", "设计", "采购", "物流", "品控", "供应链",
    "战略", "投资", "审计", "合规", "公关", "培训", "招聘", "薪酬", "绩效", "品牌",
    "渠道", "售前", "售后", "技术支持", "项目管理", "架构", "基础设施", "质量", "组织发展",
]

TOPICS = [
    "考勤异常处理", "远程办公管理", "外派人员管理", "实习生管理", "劳务派遣管理",
    "员工离职交接", "竞业限制", "保密协议管理", "知识产权归属", "专利申报流程",
    "商标管理", "合同用印", "印章管理", "档案管理", "公文写作规范",
    "会议管理", "差旅审批", "商务接待", "礼品管理", "供应商准入",
    "采购比价", "招标管理", "验收标准", "付款账期", "发票管理",
    "税务申报", "预算调整", "成本核算", "固定资产管理", "资产盘点",
    "设备报修", "机房管理", "网络准入", "VPN 使用", "数据备份",
    "灾备演练", "日志审计", "漏洞管理", "渗透测试", "应急响应",
    "灾后恢复", "账号权限申请", "离职账号回收", "代码提交规范", "分支管理",
    "发布回滚", "灰度发布", "监控告警", "值班排班", "故障复盘",
    "SLA 管理", "客户投诉", "工单流转", "满意度回访", "知识库维护",
    "培训学时", "内训师管理", "导师带教", "绩效申诉", "晋升答辩",
    "调岗流程", "薪酬保密", "社保公积金", "个税申报", "体检安排",
    "团建经费", "年会筹备", "办公用品", "会议室预订", "门禁权限",
    "快递收发", "车辆管理", "食堂管理", "宿舍管理", "班车安排",
    "差旅保险", "签证办理", "外事接待", "品牌使用", "对外宣传",
    "媒体采访", "舆情处理", "竞品分析", "市场调研", "渠道返点",
    "大客户折扣", "售前支持", "售后保修", "技术支持分级", "服务台工单",
    "变更管理", "配置管理", "容量规划", "压测规范", "代码评审",
    "技术债管理", "开源合规", "第三方组件", "专利奖励", "创新提案",
]

_NUM = re.compile(r"\d+")


def _shift_numbers(text: str, offset: int) -> str:
    """把文本里的所有整数整体偏移，使答案片段（含数字）不会在变体里出现。"""
    return _NUM.sub(lambda m: str(int(m.group()) + offset), text)


def _dept_variant(text: str, dept: str, offset: int) -> str:
    """生成某部门的变体：标题加部门前缀 + 数字偏移。"""
    lines = text.split("\n")
    out = []
    for line in lines:
        if line.startswith("# "):
            out.append(f"# {dept}部门{line[2:]}")
        else:
            out.append(_shift_numbers(line, offset))
    return "\n".join(out)


def gen_department_variants() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_files = sorted(BASE_DIR.glob("*.md"))
    count = 0
    for path in base_files:
        text = path.read_text(encoding="utf-8")
        for i, dept in enumerate(DEPARTMENTS):
            variant = _dept_variant(text, dept, offset=(i % 9) + 1)
            out = OUT_DIR / f"dept{i:02d}_{path.name}"
            out.write_text(variant, encoding="utf-8")
            count += 1
    print(f"部门变体: {count} 篇（{len(base_files)} 基础 × {len(DEPARTMENTS)} 部门）")
    return count


_GEN_SYSTEM = """你是企业制度文档撰写助手。请为给定主题撰写一篇简短的企业内部制度文档。
要求：
- 中文；含标题（用 # 开头）和 3-5 个条款小节（用 ## 开头）
- 内容具体，包含数字、时限、金额等细节
- 篇幅 300-500 字
- 只输出文档本身，不要任何解释或前后缀"""


async def _gen_one(topic: str, idx: int, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            content = await complete(
                [
                    {"role": "system", "content": _GEN_SYSTEM},
                    {"role": "user", "content": f"主题：{topic}"},
                ]
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] {topic}: {e}")
            return False
    if not content.strip():
        return False
    (OUT_DIR / f"llm{idx:03d}_{topic.replace(' ', '')}.md").write_text(
        content.strip() + "\n", encoding="utf-8"
    )
    return True


async def gen_llm_docs(limit: int | None = None) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topics = TOPICS[:limit] if limit else TOPICS
    sem = asyncio.Semaphore(5)  # 并发上限，避免打爆 API
    print(f"LLM 生成: {len(topics)} 篇（并发 5）...")
    results = await asyncio.gather(*[_gen_one(t, i, sem) for i, t in enumerate(topics)])
    ok = sum(1 for r in results if r)
    print(f"LLM 生成完成: {ok}/{len(topics)} 篇")
    return ok


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dept", action="store_true", help="跳过部门变体")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 生成")
    parser.add_argument("--llm-limit", type=int, default=None, help="限制 LLM 生成篇数（调试用）")
    args = parser.parse_args()

    if not args.skip_dept:
        gen_department_variants()
    if not args.skip_llm:
        await gen_llm_docs(args.llm_limit)

    files = list(OUT_DIR.glob("*.md"))
    print(f"\n合成语料共 {len(files)} 篇 → {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
