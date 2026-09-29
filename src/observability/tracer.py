"""审计日志 + OTel 埋点占位。"""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger("rag.audit")


def audit(event: str, **fields) -> None:
    """结构化审计日志：记录检索/向量化/LLM/工具调用，便于溯源。"""
    fields.update({"event": event, "ts": time.time()})
    logger.info(json.dumps(fields, ensure_ascii=False, default=str))


def audit_err(event: str, **fields) -> None:
    """**错误级**审计：用于"本该报警"的静默降级（P0-1 / P0-4）。

    ⚠️ 与 `audit()` 的区别不只是措辞：`logger.error` 会带 `ERROR` 前缀，
    于是这些事件能被既有的日志排查姿势捞到 ——
    `grep -n "ERROR\\|Traceback" logs/api.log`（见 CLAUDE.md 的日志纪律）。
    若只用 info，它们会淹没在正常审计里，等于"记了但没人看得见"。
    """
    fields.update({"event": event, "ts": time.time(), "level": "error"})
    logger.error(json.dumps(fields, ensure_ascii=False, default=str))


# OTel 占位：后期接入 OpenTelemetry 时，在此初始化 TracerProvider / Exporter。
