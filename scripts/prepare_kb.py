"""
知识库准备脚本：读取文档 → 分块 → 向量化 → 存入 Chroma
仅负责建库，不负责生成示例文档（示例文档由 generate_docs.py 生成）
"""
from dotenv import load_dotenv
load_dotenv()  # 让 HF_ENDPOINT 在模型下载前生效

import os
import sys

# 把项目根目录加入 Python 路径，让脚本能 import src 下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


def prepare_knowledge_base():
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw_docs")

    # 1. 读取所有文档
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents.append(f.read())

    if not documents:
        print("未找到任何文档！请先运行 scripts/generate_docs.py 生成示例文档。")
        return

    print(f"共加载 {len(documents)} 份文档")

    # 2. 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # 每块最多 500 字符
        chunk_overlap=50,   # 相邻块重叠 50 字符
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    all_chunks = []       # 文本
    all_metadatas = []    # 元数据
    all_sources = []      # 记录每个 chunk 来自哪个文件

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                doc_text = f.read()
            chunks = text_splitter.split_text(doc_text)
            all_chunks.extend(chunks)
            all_sources.extend([filename] * len(chunks)) # 每个chunk标记来源

    print(f"共切分为 {len(all_chunks)} 个 chunk")

    # 3. 加载 embedding 模型
    print("\n正在加载 embedding 模型...")
    embedding_model = SentenceTransformer("D:/Models/bge-small-zh-v1.5")

    class MyEmbeddingFunction:
        def embed_documents(self, texts):
            embeddings = embedding_model.encode(list(texts), show_progress_bar=False)
            return embeddings.tolist()

        def embed_query(self, text):
            embedding = embedding_model.encode([text], show_progress_bar=False)
            return embedding[0].tolist()

    # 4. 向量化 + 存入 Chroma
    print("正在向量化并存入 Chroma...")
    vector_store = Chroma.from_texts(
        texts=all_chunks,
        embedding=MyEmbeddingFunction(),
        metadatas=[{"source":src} for src in all_sources],
        persist_directory=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db"
        ),
        collection_name="product_docs",
    )

    print(f"\n知识库准备完成！共 {len(all_chunks)} 个 chunk 已存入 Chroma")
    print(f"向量数据库路径: data/chroma_db/")


if __name__ == "__main__":
    prepare_knowledge_base()
