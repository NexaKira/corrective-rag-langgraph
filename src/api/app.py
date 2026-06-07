"""
FastAPI 服务 — 提供 /ask 接口
"""
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.pipeline import rag_pipeline
from src.utils.tracing import setup_tracing

# 启动时初始化追踪
setup_tracing()

app = FastAPI(
    title="Corrective RAG Agent",
    description="基础 RAG 问答接口 (Phase 1)",
    version="0.1.0",
)


# ============================================================
# 请求 / 响应模型
# ============================================================

class AskRequest(BaseModel):
    query: str          # 用户问题
    k: int = 3          # 检索文档数，默认 3


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


# ============================================================
# 路由
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Corrective RAG Agent",
        "version": "0.1.0",
        "endpoints": {
            "POST /ask": "提问接口",
            "GET /health": "健康检查",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """RAG 问答：检索 + 生成"""
    result = rag_pipeline(query=req.query, k=req.k)
    return AskResponse(answer=result["answer"], sources=result["sources"])


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
