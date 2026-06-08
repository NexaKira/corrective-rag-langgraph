"""
RAG Pipeline：串联搜索 -> 拼 prompt -> 调用 LLM 生成答案
"""


# 初始化Deepseek客户端（兼容 OpenAI SDK）

import os
from dotenv import load_dotenv

from src.reranker.cross_encoder import rerank
from src.retrieval.fusion import multi_search
load_dotenv()
from openai import OpenAI
from src.retrieval.vector import retrieve_docs


_deepseek_client = None

def _get_llm():
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        _deepseek_client = OpenAI(
            api_key= api_key,
            base_url="https://api.deepseek.com"
        )
    return _deepseek_client

# Prompt 模板
SYSTEM_PROMPT = """你是一个知识库问答助手。请根据以下参考资料回答用户问题。
要求：
1. 如果参考资料足以回答问题，请给出详细的答案
2. 如果参考资料不足以回答问题，请如实说明"根据现有资料无法回答"，不要编造
3. 回答时请用中文"""

def _build_messages(query: str, docs: list[dict]) -> list[dict]:
    """把检索到的文档拼进 prompt"""
    # 拼接所有检索到的文档
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[参考资料 {i}](来源: {doc['source']})\n{doc['content']}")
    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料:\n{context}\n\n用户问题: {query}\n\n请回答:"}
    ]

# 对外暴露的唯一接口
def rag_pipeline(query: str, k: int = 3):
    # 1.检索
    #docs = retrieve_docs(query, k)

    #第一步：多路召回（BM25 + 向量）-> RRF融合
    candidates = multi_search(query, k)

    #第二步：Cross-Encoder 精排 -> 取 Top-K
    docs = rerank(query, candidates, top_k=k)

    # 2.拼 prompt
    messages = _build_messages(query, docs)

    # 3.调 LLM 生成
    client = _get_llm()
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {
        "answer":answer,
        "sources":docs
    }

# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    query = "蓝牙耳机怎么配对"
    print(f"用户问题: {query}\n")

    result = rag_pipeline(query)

    print(f"AI 回答:\n{result['answer']}\n")
    print(f"参考来源 ({len(result['sources'])} 条):")

    for s in result["sources"]:
        print(f"  - {s['source']}")
