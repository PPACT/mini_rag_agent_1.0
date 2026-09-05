"""Mock MCP 服务（模拟企业内部业务工具：员工信息查询）。

上线后删除此文件，替换为企业真实 MCP 服务地址（在 mcp_client_wrapper.py 里改连接即可）。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("employee_info")

_MOCK_DB = {
    "E001": {"name": "张三", "department": "IT", "position": "后端工程师"},
    "E002": {"name": "李四", "department": "HR", "position": "招聘专员"},
    "E003": {"name": "王五", "department": "财务", "position": "会计"},
}


@mcp.tool()
async def query_employee_info(employee_id: str) -> str:
    """根据工号查询员工信息（姓名/部门/职位）。"""
    emp = _MOCK_DB.get(employee_id)
    if emp is None:
        return f"未找到工号 {employee_id} 的员工"
    return f"工号 {employee_id}：{emp['name']}，部门 {emp['department']}，职位 {emp['position']}"


if __name__ == "__main__":
    mcp.run()  # stdio 传输
