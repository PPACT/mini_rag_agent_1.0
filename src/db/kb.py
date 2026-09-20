"""知识库（kb）标识 —— **两库分离**：真实业务库 / 压测库。

背景（详见 `docs/交流区.md` §1.6）：
合成评测语料里有 624 篇**刻意造出的近重复干扰**（同一个制度几十份微差版本）。
若与真实文档同库，用户问真实业务问题时 top-5 会被这些干扰版本淹没 → **真实答案排不上**。
故两库物理隔离（两个容器），本模块是它们的唯一标识来源。

⚠️ **kb 是必填的一等信息，不设默认值**：
一旦有默认（如 `kb=None → 真实库`），"忘传"就会**静默落到真实库**——
压测数据可能被灌进真实库，且不报错。本项目反复踩的正是这类"静默降级"。
把它做成必填参数，忘传就**报错**（fail-closed），而不是静默走错库。
"""
from __future__ import annotations

KB_REAL = "real"        # 真实业务库（/chat 默认入口）
KB_STRESS = "stress"    # 压测库（合成语料；离线可选环境）

VALID_KBS: tuple[str, ...] = (KB_REAL, KB_STRESS)


class UnknownKBError(ValueError):
    """kb 取值非法。

    ⚠️ 必须**抛错**，绝不回退到某个默认库——回退就是静默走错库。
    """


def validate(kb: str) -> str:
    """校验 kb 取值；非法即抛错（fail-closed）。"""
    if kb not in VALID_KBS:
        raise UnknownKBError(
            f"未知的知识库 kb={kb!r}；合法值：{VALID_KBS}。"
            f"（不提供默认值：静默回退会导致数据灌错库）"
        )
    return kb
