"""
基于 LangSmith，会自动追踪所有 LangChain 组件
"""
import os

def setup_tracing():
    """
    初始化 LangSmith 追踪。
    调用此函数后，所有 LangChain 组件的调用会自动记录到 LangSmith。

    需要的环境变量（在 .env 中配置）：
        LANGSMITH_API_KEY   — 从 https://smith.langchain.com 获取（免费）
        LANGSMITH_PROJECT   — 项目名称，用于区分不同实验
    """
    langsmith_key = os.getenv("LANGSMITH_API_KEY")

    if not langsmith_key:
        print("[追踪] 未检测到 LANGSMITH_API_KEY，跳过 LangSmith 初始化")
        print("[追踪] 如需追踪，请在 .env 中设置 LANGSMITH_API_KEY")
        return
    
    # LangChain 会自动读取这些环境变量，不需要额外代码
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", "corrective-rag-phase1")

    print(f"[追踪] LangSmith 已启用，项目: {os.environ['LANGSMITH_PROJECT']}")