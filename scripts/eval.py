"""
评估脚本：对比 Phase 1（单路向量）和 Phase 2（多路召回+精排）的检索质量
指标：Recall@5, MRR@10
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.vector import retrieve_docs as vector_only
from src.retrieval.fusion import multi_search
from src.reranker.cross_encoder import rerank


def load_test_queries():
    """加载标注测试集"""
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eval", "test_queries.json"
    )
    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_sources, relevant_sources, k=5):
    """
    Recall@K: 前 K 个结果中命中了多少相关文档 / 总相关文档数
    衡量"召回是否全面"
    """
    top_k = set(retrieved_sources[:k])
    relevant = set(relevant_sources)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)


def mrr_at_k(retrieved_sources, relevant_sources, k=10):
    """
    MRR@K: 第一个相关文档排在第几位（倒数平均）
    衡量"第一个正确答案好不好找"
    """
    relevant = set(relevant_sources)
    for i, src in enumerate(retrieved_sources[:k], start=1):
        if src in relevant:
            return 1.0 / i
    return 0.0


def evaluate(name, search_fn):
    """
    跑一轮评估

    参数:
        name: 评估名称（如 "Phase 1: 单路向量"）
        search_fn: 检索函数，签名为 (query) -> list[dict]
    """
    queries = load_test_queries()

    total_recall_5 = 0.0
    total_mrr_10 = 0.0
    total = len(queries)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for q in queries:
        results = search_fn(q["query"])
        sources = [r["source"] for r in results]

        r5 = recall_at_k(sources, q["relevant_docs"], k=5)
        m10 = mrr_at_k(sources, q["relevant_docs"], k=10)

        total_recall_5 += r5
        total_mrr_10 += m10

        status = "✓" if r5 == 1.0 else "✗"
        print(f"  {status} 召回={r5:.0f}  MRR={m10:.2f}  |  {q['query']}")

    print(f"\n  {'─'*50}")
    print(f"  平均 Recall@5: {total_recall_5/total:.2%}")
    print(f"  平均 MRR@10:   {total_mrr_10/total:.2%}")
    print()

    return total_recall_5 / total, total_mrr_10 / total


# ============================================================
# Phase 1 vs Phase 2
# ============================================================

def phase1_search(query):
    """Phase 1: 仅向量检索"""
    return vector_only(query, k=10)


def phase2_search(query):
    """Phase 2: 多路召回 + 精排"""
    candidates = multi_search(query, k=10)
    return rerank(query, candidates, top_k=10)


if __name__ == "__main__":
    r1, m1 = evaluate("Phase 1: 单路向量检索", phase1_search)
    r2, m2 = evaluate("Phase 2: 多路召回 + RRF + Cross-Encoder", phase2_search)

    print("=" * 60)
    print("  对比总结")
    print("=" * 60)
    print(f"  Recall@5:  {r1:.2%} → {r2:.2%}  ({r2-r1:+.1%})")
    print(f"  MRR@10:    {m1:.2%} → {m2:.2%}  ({m2-m1:+.1%})")
    print()
