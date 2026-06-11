"""
网络搜索工具 —— 使用 Tavily API
需要 .env 中设置 TAVILY_API_KEY（可选，没有 Key 也能运行空壳）
"""
import json
import os
from src.tools.base import BaseTool
from langchain_tavily import TavilySearch


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "搜索互联网获取最新信息。适用于知识库中没有的、需要最新数据的问题"
    parameters = {
        "query": {
            "type": "string",
            "description": "搜索查询词，如 '氮化镓充电器 2026 新品'",
        }
    }
    is_dangerous = False

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "错误: 缺少搜索查询"

        api_key = os.getenv("TAVILY_API_KEY", "")

        if not api_key:
            return (
                "[模拟搜索结果] 未配置 TAVILY_API_KEY，无法真实搜索。\n"
                f"搜索词: {query}\n"
                "提示: 在 .env 中设置 TAVILY_API_KEY 即可启用真实搜索。\n"
                "免费注册: https://tavily.com"
            )

        try:
            # 使用最新的 TavilySearch 类，指定最多返回 3 条结果
            # 它会自动读取系统中的 TAVILY_API_KEY 环境变量
            langchain_tavily = TavilySearch(max_results=3)
            # LangChain 会自动把结果格式转化为字符串
            raw_result = langchain_tavily.run(query)
            search_results = raw_result.get("results", [])
            final_string = json.dumps(search_results, ensure_ascii=False)
            return final_string if final_string else "未找到相关搜索结果"
        except Exception as e:
            return f"搜索出错: {e}"
