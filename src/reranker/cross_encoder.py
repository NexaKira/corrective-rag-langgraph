"""
Cross-Encoder 精排模块
对多路召回融合后的候选文档做精准重排序
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sentence_transformers import CrossEncoder

# ============================================================
# 模型单例
# ============================================================

_reranker = None


def _get_reranker():
    """加载 Cross-Encoder 模型（单例）"""
    global _reranker
    if _reranker is None:
        # 中文推荐 BAAI/bge-reranker-base（~1.1GB），效果好
        # 轻量替代: BAAI/bge-reranker-v2-m3（~560MB，多语言）
        # 如果下载困难，可以像 embedding 模型一样放到本地路径
        model_name = "BAAI/bge-reranker-base"
        print(f"[精排] 正在加载 Cross-Encoder 模型: {model_name}")
        _reranker = CrossEncoder(model_name)
        print("[精排] 模型加载完成")
    return _reranker


# ============================================================
# 对外接口
# ============================================================

def rerank(query: str, candidates: list[dict], top_k: int = 3):
    """
    对候选文档精排，返回最相关的 top_k 篇

    参数:
        query: 用户问题
        candidates: 候选文档列表 [{"content": "...", "source": "..."}, ...]
        top_k: 最终返回的文档数

    返回:
        list[dict]: 精排后的文档，附带 cross_score 字段
    """
    if len(candidates) <= top_k:
        # 候选不多，无需精排
        for c in candidates:
            c["cross_score"] = None
        return candidates[:top_k]

    model = _get_reranker()

    # 构造 (query, doc_content) 对
    pairs = [(query, doc["content"]) for doc in candidates]

    # 批量推理：一次过模型，比逐个快很多
    scores = model.predict(pairs, show_progress_bar=False)

    # 分数附加到文档上
    for doc, score in zip(candidates, scores):
        doc["cross_score"] = float(score)

    # 按分数降序排列，取 top_k
    sorted_candidates = sorted(candidates, key=lambda d: d["cross_score"], reverse=True)
    return sorted_candidates[:top_k]


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    from src.retrieval.fusion import multi_search

    query = "蓝牙耳机怎么配对"
    print(f"查询: {query}\n")

    # 1. 多路召回 + RRF 融合
    fused = multi_search(query, k=5)
    print(f"\nRRF 融合后共 {len(fused)} 个候选\n")

    # 2. Cross-Encoder 精排
    final = rerank(query, fused, top_k=3)

    print(f"精排后取 Top-{len(final)}:")
    for i, doc in enumerate(final, 1):
        score_str = f"{doc['cross_score']:.4f}" if doc['cross_score'] else "N/A"
        print(f"  {i}. [{score_str}] {doc['source']}: {doc['content'][:80]}...")
