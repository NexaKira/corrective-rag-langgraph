"""
Agent 图构建器 —— 包含工具调用 + Corrective RAG 的完整 Agent 图
只负责构建图和路由逻辑，节点和 State 来自 state.py / nodes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.graph.nodes import (
    plan_node, tool_node, agent_answer_node,
    retrieve_node, grade_node, rewrite_node, generate_node,
)
from src.tools.registry import register_default_tools


# ============================================================
# 路由函数
# ============================================================

def route_after_plan(state: AgentState) -> str:
    action = state.get("_plan_action", "answer")
    if action == "tool":
        if state.get("tool_round", 0) >= state.get("max_tool_rounds", 2):
            return "answer"
        return "tool"
    if action == "knowledge":
        return "knowledge"
    return "answer"


def route_after_grade(state: AgentState) -> str:
    history = state.get("retrieval_history", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if history and history[-1].get("grade") == "adequate":
        return "generate"
    if iteration >= max_iter:
        return "generate"
    return "rewrite"


# ============================================================
# 构建图
# ============================================================

def build_agent_graph():
    graph = StateGraph(AgentState)

    # 注册节点（全部来自 nodes.py）
    graph.add_node("plan", plan_node)
    graph.add_node("tool", tool_node)
    graph.add_node("agent_answer", agent_answer_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    # 普通边
    graph.add_edge(START, "plan")
    graph.add_edge("tool", "plan")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("agent_answer", END)

    # 条件边
    graph.add_conditional_edges("plan", route_after_plan, {
        "tool": "tool", "knowledge": "retrieve", "answer": "agent_answer",
    })
    graph.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate", "rewrite": "rewrite",
    })

    return graph.compile()


# ============================================================
# 便捷调用
# ============================================================

_agent_graph = None

def run_agent(query: str, max_tool_rounds: int = 2, max_iterations: int = 3) -> dict:
    global _agent_graph
    if _agent_graph is None:
        register_default_tools()
        _agent_graph = build_agent_graph()

    return _agent_graph.invoke({
        "original_query": query,
        "tool_history": [],
        "tool_round": 0,
        "max_tool_rounds": max_tool_rounds,
        "_plan_action": "",
        "_plan_tool_name": "",
        "_plan_tool_input": {},
        "current_query": query,
        "iteration": 0,
        "documents": [],
        "retrieval_history": [],
        "max_iterations": max_iterations,
        "answer": "",
    })
