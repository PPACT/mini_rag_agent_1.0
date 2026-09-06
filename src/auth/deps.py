"""鉴权依赖：从 Authorization header 解析用户身份。"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException


@dataclass(frozen=True)
class User:
    """当前登录用户（服务端身份，不可由客户端伪造）。"""

    name: str
    department: str
    secret_level: int


# 演示用户表（token -> 身份）。真实系统替换为 DB 查询 / JWT 校验 / SSO。
DEMO_USERS: dict[str, User] = {
    "demo-it-token": User(name="张三", department="IT", secret_level=3),
    "demo-hr-token": User(name="李四", department="HR", secret_level=2),
    "demo-public-token": User(name="访客", department="公开", secret_level=0),
}


async def get_current_user(authorization: str | None = Header(default=None)) -> User:
    """校验 `Authorization: Bearer <token>`，返回当前用户；失败抛 401。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少鉴权信息（Authorization: Bearer <token>）")
    token = authorization[len("Bearer "):].strip()
    user = DEMO_USERS.get(token)
    if user is None:
        raise HTTPException(status_code=401, detail="无效 token")
    return user
