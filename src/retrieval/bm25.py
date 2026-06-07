"""
BM25 关键词检索模块
基于 rank_bm25 + jieba 分词，提供传统关键词召回能力
"""
import os
import sys
import pickle
import jieba
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# BM25 索引（懒加载单例）
# ============================================================

_bm25_index = None
_corpus = None       # 保存原始文本，用于返回检索结果
_corpus_sources = None  # 保存每个 chunk 的来源


def _load_or_build_index():
    """加载或构建 BM25 索引"""
    global _bm25_index, _corpus, _corpus_sources

    if _bm25_index is not None:
        return

    # 尝试从缓存加载（索引构建很快，但分词还是有点开销）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_path = os.path.join(project_root, "data", "bm25_index.pkl")
    docs_dir = os.path.join(project_root, "data", "raw_docs")

    # ---------- 加载文档 ----------
    documents = []
    sources = []
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
                doc_text = f.read()
            chunks = text_splitter.split_text(doc_text)
            documents.extend(chunks)
            sources.extend([filename] * len(chunks))

    # ---------- 分词 ----------
    print(f"[BM25] 正在对 {len(documents)} 个 chunk 分词...")
    tokenized = [list(jieba.cut(doc)) for doc in documents]

    # ---------- 构建索引 ----------
    _bm25_index = BM25Okapi(tokenized)
    _corpus = documents
    _corpus_sources = sources

    # 缓存到磁盘（下次启动更快）
    with open(cache_path, "wb") as f:
        pickle.dump({
            "tokenized": tokenized,
            "corpus": documents,
            "sources": sources,
        }, f)

    print(f"[BM25] 索引构建完成，{len(documents)} 个文档")


# ============================================================
# 对外接口
# ============================================================

def bm25_search(query: str, k: int = 10):
    """
    BM25 关键词检索

    返回:
        list[dict]: [{"content": "...", "source": "...", "score": 1.23}, ...]
        score 越高表示越相关
    """
    _load_or_build_index()

    # 分词
    tokenized_query = list(jieba.cut(query))

    # 检索（返回得分和索引）
    scores = _bm25_index.get_scores(tokenized_query)

    # 取 top-k
    # BM25Okapi 没有内置的 top-k，需要自己排序
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # 过滤掉完全不相关的
            results.append({
                "content": _corpus[idx],
                "source": _corpus_sources[idx],
                "score": float(scores[idx]),
            })

    return results


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    query = "蓝牙耳机怎么配对"
    print(f"查询: {query}\n")

    docs = bm25_search(query, k=5)

    for i, doc in enumerate(docs, 1):
        print(f"--- BM25 结果 {i} (来源: {doc['source']}, 分数: {doc['score']:.2f}) ---")
        print(doc["content"][:150] + "...")
        print()
