"""
多路召回融合模块
RRF (Reciprocal Rank Fusion) 算法 + 统一检索入口
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.vector import retrieve_docs as vector_search
from src.retrieval.bm25 import bm25_search


# ============================================================
# RRF 融合算法
# ============================================================

def rrf_fusion(result_lists, k=60):
    """
    对多路召回结果做 RRF 融合

    参数:
        result_lists: list[list[dict]] — 每路召回的结果列表
            每个 dict 必须包含 "content" 和 "source" 字段
        k: RRF 常数（默认 60），平衡排名权重

    返回:
        list[dict]: 融合后的结果，按 RRF 分数降序排列

    RRF 公式: score(doc) = Σ 1 / (k + rank_i)
    其中 rank_i 是文档在第 i 路召回中的排名（从 1 开始）
    """
    from collections import defaultdict

    # 用 (source, content前50字) 作为文档唯一标识
    # 因为同一个 chunk 可能在不同路中以不同排名出现
    def _make_key(doc):
        return (doc["source"], doc["content"][:50])

    # 累加 RRF 分数
    scores = defaultdict(float)
    doc_map = {}  # key → 原始 doc 信息

    for result in result_lists:
        for rank, doc in enumerate(result, start=1):
            key = _make_key(doc)
            scores[key] += 1 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    # 按 RRF 分数降序排列
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

    fused = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = scores[key]
        fused.append(doc)

    return fused


# ============================================================
# 统一检索接口（Phase 2 的核心）
# ============================================================

def multi_search(query: str, k: int = 10):
    """
    多路召回（BM25 + 向量）→ RRF 融合

    参数:
        query: 用户问题
        k: 每路召回的数量（融合后结果 ≤ 2*k）

    返回:
        list[dict]: 融合排序后的文档列表
    """
    print(f"[检索] 正在多路召回: {query}")

    # 第 1 路：BM25 关键词
    bm25_results = bm25_search(query, k=k)

    # 第 2 路：向量语义
    vector_results = vector_search(query, k=k)

    # RRF 融合
    fused = rrf_fusion([bm25_results, vector_results])

    print(f"[检索] BM25: {len(bm25_results)} | 向量: {len(vector_results)} | 融合: {len(fused)}")

    return fused


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    query = "蓝牙耳机怎么配对"
    print(f"查询: {query}\n")

    results = multi_search(query, k=5)

    for i, doc in enumerate(results, 1):
        print(f"--- 融合结果 {i} (来源: {doc['source']}, RRF: {doc['rrf_score']:.4f}) ---")
        print(doc["content"][:150] + "...")
        print()
