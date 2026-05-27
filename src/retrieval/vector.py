"""
Chroma 向量检索模块
提供语义搜索能力：输入用户问题，返回最相关的文档片段
"""

import os

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma


_embedding_model = None
_vector_store = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("D:/Models/bge-small-zh-v1.5")
    return _embedding_model

class MyEmbeddingFunction:
    """适配LangChain Chroma的embedding接口"""
    def embed_documents(self, texts):
        model = _get_embedding_model()
        return model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text):
        model = _get_embedding_model()
        return model.encode([text], show_progress_bar=False)[0].tolist()
    
def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        # 项目根目录 -> data/chroma_db/
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        persist_dir = os.path.join(project_root, "data", "chroma_db")

        if not os.path.exists(persist_dir):
            raise FileNotFoundError(
                f"向量库目录不存在：{persist_dir}\n"
                "请先运行：python scripts/prepare_kb.py"
            )
        
        _vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=MyEmbeddingFunction(),
            collection_name="product_docs"
        )
    return _vector_store

# 对外暴露唯一接口
def retrieve_docs(query: str, k: int=3):
    """
    参数：
        query：用户问题
        k：返回前k个最相关结果，默认3

    返回：
        list[dict]
    """
    vector_store = _get_vector_store()
    results = vector_store.similarity_search_with_score(query, k)
    
    docs = []
    for doc, score in results:
        docs.append({
            "content":doc.page_content,
            "source":doc.metadata.get("source", "unknown")
        })

    return docs

# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    query = "蓝牙耳机怎么配对"
    print(f"查询{query}\n")

    docs = retrieve_docs(query, k=3)

    for i, doc in enumerate(docs, 1):
        print(f"--- 结果 {i} (来源: {doc['source']}) ---")
        print(doc["content"][:200] + "...")
        print()