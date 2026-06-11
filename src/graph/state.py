"""
LangGraph State 定义 —— Corrective RAG 的状态数据结构
"""
from typing import TypedDict


class RagState(TypedDict):
    # ── 不变的 ──
    original_query: str          # 用户最初的问题，全程不变
    max_iterations: int          # 最多检索几轮（默认 3）

    # ── 每轮变化的 ──
    current_query: str           # 当前这轮要检索的 query（rewrite 会改它）
    iteration: int               # 当前是第几轮（从 0 开始）
    documents: list[dict]        # 最新一轮检索到的文档

    # ── 累积的 ──
    # Annotated[list, operator.add] 表示：节点返回的 list 会和原有 list 拼接
    # 而不是覆盖！这就是你 app.py 里 messages 自动追加的原理
    retrieval_history: list[dict]

    # ── 最终的 ──
    answer: str                  # generate 节点产出，初始为空


class AgentState(TypedDict):
    """Agent State —— 包含工具调用 + Corrective RAG 的完整状态"""
    # 用户输入
    original_query: str

    # 工具调用
    tool_history: list[dict]
    tool_round: int
    max_tool_rounds: int

    # 图内部路由（临时字段，下划线前缀）
    _plan_action: str
    _plan_tool_name: str
    _plan_tool_input: dict

    # ── 以下字段传给 Phase 3 节点 ──
    current_query: str
    iteration: int
    documents: list[dict]
    retrieval_history: list[dict]
    max_iterations: int

    # 最终输出
    answer: str

