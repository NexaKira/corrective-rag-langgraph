"""
构建 Corrective RAG 的 LangGraph 图
包含 START → retrieve → grade → [条件] → generate/rewrite → END
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, START, END
from src.graph.state import RagState
from src.graph.nodes import retrieve_node, grade_node, rewrite_node, generate_node


# ============================================================
# 条件边的路由函数（Phase 3 的灵魂）
# ============================================================

def should_continue(state: RagState) -> str:
    """
    grade 节点执行完后，LangGraph 调用这个函数决定下一步去哪。

    返回值必须是已注册的节点名字符串

    三种分支：
      "generate" — 文档够好，直接生成答案
      "rewrite"  — 文档不够好且还没到上限，重写查询
      "generate" — 到了上限，即使不够好也得生成（兜底）
    """
    history = state.get("retrieval_history", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    # 检查最后一轮的评估结果
    if history:
        last_grade = history[-1].get("grade")
        if last_grade == "adequate":
            print(f"\n  ✅ 文档充足，准备生成答案")
            return "generate"

    # 达到上限 → 强制生成
    if iteration >= max_iter:
        print(f"\n  ⚠️ 已达最大迭代次数({max_iter})，强制生成答案")
        return "generate"

    # 文档不够 + 没到上限 → 重写查询
    print(f"\n  🔄 文档不足 ({iteration}/{max_iter})，重写查询...")
    return "rewrite"


# ============================================================
# 构建图
# ============================================================

def build_graph():
    """构建并编译 Corrective RAG 图"""
    graph = StateGraph(RagState)

    # 1. 注册节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    # 2. 普通边（固定路线）
    graph.add_edge(START, "retrieve")     # 入口 → 检索
    graph.add_edge("retrieve", "grade")   # 检索 → 评估
    graph.add_edge("rewrite", "retrieve") # 重写 → 回到检索（闭环！）
    graph.add_edge("generate", END)       # 生成 → 出口

    # 3. 条件边（Phase 3 最关键的语法！）
    graph.add_conditional_edges(
        "grade",             # 从 grade 节点出发
        should_continue,     # 调用这个函数决定走哪条路
        {
            "generate": "generate",  # 返回 "generate" → 走向 generate 节点
            "rewrite": "rewrite",    # 返回 "rewrite"  → 走向 rewrite 节点
        },
    )

    # 4. 编译
    return graph.compile()


# ============================================================
# 单例
# ============================================================

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ============================================================
# 便捷调用
# ============================================================

def run_corrective_rag(query: str, max_iterations: int = 3) -> dict:
    """
    运行 Corrective RAG，返回最终 state

    参数:
        query: 用户问题
        max_iterations: 最大检索轮次，默认 3

    返回:
        RagState: 包含 answer、retrieval_history 等所有字段
    """
    graph = get_graph()

    initial_state = {
        "original_query": query,
        "current_query": query,
        "iteration": 0,
        "documents": [],
        "retrieval_history": [],
        "max_iterations": max_iterations,
        "answer": "",
    }

    return graph.invoke(initial_state)
