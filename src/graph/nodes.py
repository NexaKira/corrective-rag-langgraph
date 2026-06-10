"""
Corrective RAG 的 4 个节点函数
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from src.retrieval.fusion import multi_search
from src.reranker.cross_encoder import rerank


# ============================================================
# LLM 客户端
# ============================================================

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
    return _llm


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    client = _get_llm()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content


def _parse_json(text: str) -> dict:
    """安全解析 LLM 返回的 JSON"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ============================================================
# 节点 1: retrieve
# ============================================================

def retrieve_node(state: dict) -> dict:
    query = state["current_query"]
    iteration = state["iteration"] + 1

    print(f"\n{'─'*40}")
    print(f"  📥 [第 {iteration} 轮检索] query: {query}")
    print(f"{'─'*40}")

    candidates = multi_search(query, k=10)
    docs = rerank(query, candidates, top_k=3)

    for i, doc in enumerate(docs):
        score_info = f" (cross: {doc.get('cross_score', 0):.4f})" if doc.get('cross_score') else ""
        print(f"    {i+1}. {doc['source']}{score_info}")

    # 手动追加历史记录（因为 state 是普通 list，不会自动追加）
    new_history = state.get("retrieval_history", []) + [{
        "query": query,
        "docs": docs,
        "grade": None,
    }]

    return {
        "current_query": query,
        "iteration": iteration,
        "documents": docs,
        "retrieval_history": new_history,
    }


# ============================================================
# 节点 2: grade
# ============================================================

GRADE_PROMPT = """你是一个检索质量评估专家。判断检索到的文档能否回答用户问题。

输出格式：只输出一个 JSON，不要其他内容。
{"grade": "adequate", "reason": "原因"}
或
{"grade": "inadequate", "reason": "原因"}"""


def grade_node(state: dict) -> dict:
    query = state["original_query"]
    docs = state["documents"]

    print(f"\n  🔍 [质量评估] 正在判断...")

    docs_text = "\n\n---\n".join([
        f"[文档{i+1}] {d['source']}\n{d['content'][:300]}"
        for i, d in enumerate(docs)
    ])

    user_prompt = f"用户问题: {query}\n\n检索到的文档:\n{docs_text}\n\n请判断，输出 JSON。"

    result_text = _call_llm(GRADE_PROMPT, user_prompt)
    result = _parse_json(result_text)

    grade = result.get("grade", "adequate")
    reason = result.get("reason", "")

    print(f"  📊 评估: {grade}")
    print(f"  💬 {reason}")

    # 更新最后一条历史记录的 grade
    history = state.get("retrieval_history", [])
    if history:
        history[-1]["grade"] = grade

    return {"retrieval_history": history}


# ============================================================
# 节点 3: rewrite
# ============================================================

REWRITE_PROMPT = """你是一个搜索查询优化专家。最初的查询没有找到足够的信息，
请改写查询，使其更可能找到相关文档。

改写策略：
1. 提取原始问题中的核心实体和关键词
2. 用同义词或相关术语扩展搜索范围
3. 去掉原始查询中被证明无效的限定词
4. 如果知道缺少哪类信息，明确加入这类信息的关键词

只输出改写后的查询，不要加引号或解释。"""


def rewrite_node(state: dict) -> dict:
    query = state["original_query"]
    history = state.get("retrieval_history", [])

    print(f"\n  ✏️  [查询重写] 正在生成更好的查询...")

    # 把之前的失败尝试也告诉 LLM
    prev_attempts = "\n".join([
        f"尝试{entry['query']}: 结果{entry['grade']}"
        for entry in history if entry.get('query')
    ])

    user_prompt = f"""原始问题: {query}

之前的搜索尝试:
{prev_attempts}

请改写一个更好的搜索查询："""

    new_query = _call_llm(REWRITE_PROMPT, user_prompt).strip()
    # 清理可能的引号
    new_query = new_query.strip('"').strip("'").strip()
    # 有时 LLM 会在前面加 "改写后的查询："，去掉
    if "：" in new_query:
        new_query = new_query.split("：", 1)[-1].strip()

    print(f"  📝 新查询: {new_query}")

    return {"current_query": new_query}


# ============================================================
# 节点 4: generate
# ============================================================

GENERATE_PROMPT = """你是一个知识库问答助手。请根据参考资料回答用户问题。
如果参考资料不足以完全回答问题，基于已有信息给出最佳回答，
并明确指出哪些部分在资料中没有找到。"""


def generate_node(state: dict) -> dict:
    query = state["original_query"]
    history = state.get("retrieval_history", [])

    print(f"\n  🤖 [生成答案] 正在综合所有轮次的文档...")

    # 收集所有轮次的文档，去重
    seen = set()
    all_docs = []
    for entry in history:
        for doc in entry.get("docs", []):
            key = (doc["source"], doc["content"][:50])
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    print(f"  📚 去重后共 {len(all_docs)} 份文档")

    # 拼接文档
    context = "\n\n".join([
        f"[参考 {i+1}] 来源: {d['source']}\n{d['content']}"
        for i, d in enumerate(all_docs)
    ])

    user_prompt = f"参考资料:\n{context}\n\n用户问题: {query}\n\n请回答:"

    answer = _call_llm(GENERATE_PROMPT, user_prompt)

    # 打印迭代总结
    print(f"\n  {'─'*40}")
    print(f"  📊 检索总结: 共 {state['iteration']} 轮")
    for i, entry in enumerate(history, 1):
        print(f"     第{i}轮: \"{entry['query'][:40]}...\" → {entry['grade']}")
    print(f"  {'─'*40}")

    return {"answer": answer}
