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


# OTel 占位：后期接入 OpenTelemetry 时，在此初始化 TracerProvider / Exporter。
