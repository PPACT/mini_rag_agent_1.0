"""对拍 .env.example 与 settings.py 的默认值，找出不一致（`eval/` 下，已纳入版本库——**证据脚本不能放会被清理的 `logs/`**）

背景：已发现两处同类问题——
  · RERANK_BACKEND      ：示例缺失，代码默认 llm，而演示靠 start_demo 才走 local
  · QUERY_REWRITE_ENABLED：示例写 true，代码默认 false，且实测"开着有损"
这类不一致会让"照示例复制 .env"的人拿到与代码默认不同的行为。
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.config.settings import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
example = {}
for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    example[k.strip().upper()] = v.strip()

# 从 .env.example 复制出的 .env 会得到这些值 → 与代码默认值对比
print(f"{'配置项':<26}{'.env.example':<22}{'settings 默认':<20}{'判定'}")
print("-" * 80)
diff = 0
for name, f in Settings.model_fields.items():
    key = name.upper()
    default = f.default
    if key in example:
        shown = example[key]
        same = str(default).lower() == shown.lower()
        verdict = "✅ 一致" if same else "⚠️ 不一致"
        if not same:
            diff += 1
        print(f"{key:<26}{shown:<22}{str(default):<20}{verdict}")
    else:
        diff += 1
        print(f"{key:<26}{'（示例未列出）':<22}{str(default):<20}{'⚠️ 缺失'}")

print(f"\n共 {diff} 项需要注意。")
print("判据：不一致/缺失会让『照 .env.example 复制』的部署与代码默认行为不同 —— 要么对齐，要么加注释说明。")
