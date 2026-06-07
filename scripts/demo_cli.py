"""
命令行交互式 RAG 测试
直接输入问题，回车看结果，输入 quit 退出
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.pipeline import rag_pipeline


def main():
    print("=" * 50)
    print("  Corrective RAG Agent — CLI 交互测试")
    print("  输入问题后回车，输入 quit 退出")
    print("=" * 50)

    while True:
        try:
            query = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print("🤖 AI 思考中...")
        result = rag_pipeline(query)

        print(f"\n🤖 AI: {result['answer']}")
        print(f"\n📚 参考来源:")
        for s in result["sources"]:
            print(f"   📄 {s['source']}")


if __name__ == "__main__":
    main()
