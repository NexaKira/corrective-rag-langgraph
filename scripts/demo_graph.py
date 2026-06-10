"""
Corrective RAG 交互测试 —— 观察多轮检索 + 查询重写的完整过程
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.builder import run_corrective_rag


def main():
    print("=" * 50)
    print("  Corrective RAG Agent — 多轮检索测试")
    print("  可以观察: 检索 → 评估 → 重写 → 再检索")
    print("=" * 50)

    while True:
        try:
            query = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 再见!")
            break

        print("🔄 Agent 思考中...")
        result = run_corrective_rag(query, max_iterations=3)

        print(f"\n{'='*50}")
        print(f"🤖 AI: {result['answer']}")
        print(f"\n📊 检索统计: 共 {result['iteration']} 轮")
        for i, entry in enumerate(result.get("retrieval_history", []), 1):
            grade = entry.get("grade", "?")
            docs_count = len(entry.get("docs", []))
            print(f"   第{i}轮: \"{entry['query'][:50]}\" → {grade} ({docs_count}篇)")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
