"""
工具注册表 —— 管理所有可用工具
"""
from src.tools.calculator import CalculatorTool
from src.tools.web_search import WebSearchTool


# 全局注册表
_tools: dict[str, object] = {}


def register_default_tools():
    """注册默认工具（启动时调用一次）"""
    _tools.clear()

    calc = CalculatorTool()
    _tools[calc.name] = calc

    web = WebSearchTool()
    _tools[web.name] = web


def get_tool(name: str):
    """根据名称获取工具"""
    return _tools.get(name)


def list_tools():
    """列出所有工具"""
    return list(_tools.values())


def build_tools_prompt() -> str:
    """生成给 LLM 看的工具列表 prompt"""
    if not _tools:
        register_default_tools()

    lines = ["可用工具列表："]
    for tool in _tools.values():
        lines.append(tool.to_prompt_description())
    return "\n".join(lines)
