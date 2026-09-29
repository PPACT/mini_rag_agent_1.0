"""MCP 客户端封装（langchain-mcp-adapters）。"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection

_MOCK_SERVER = Path(__file__).resolve().parent / "mcp_mock_server.py"

_client: MultiServerMCPClient | None = None


async def open_mcp() -> None:
    """初始化 MCP 客户端（应用 startup 时调用）。上线替换真实 MCP 服务只改这里的连接。"""
    global _client
    connections = {
        "employee_info": StdioConnection(
            transport="stdio",
            command=sys.executable,
            args=[str(_MOCK_SERVER)],
        ),
    }
    # 注意：0.1.0+ 不能再当 async context manager 用，直接构造即可
    _client = MultiServerMCPClient(connections)


async def close_mcp() -> None:
    """清理（会话随进程退出释放）。"""
    global _client
    _client = None


async def get_tools() -> list:
    """返回 MCP 工具列表（LangChain 工具）。"""
    if _client is None:
        raise RuntimeError("MCP 客户端未初始化")
    return await _client.get_tools()
