"""
全功能 Agent 测试：工具调用 + Corrective RAG
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.agent_builder import run_agent


def main():
    print("=" * 55)
    print("  🤖 Corrective RAG Agent — 全功能测试")
    print("  工具: 计算器 | 网络搜索 | 知识库检索")
    print("  输入 quit 退出")
    print("=" * 55)

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
        result = run_agent(query)

        print(f"\n{'='*55}")
        print(f"🤖 AI: {result['answer']}")

        # 工具调用总结
        tool_history = result.get("tool_history", [])
        if tool_history:
            print(f"\n🔧 工具调用记录 ({len(tool_history)} 次):")
            for t in tool_history:
                print(f"   • {t['tool']}: {t['input']} → {t['output'][:100]}")

        # 检索总结
        retrieval_history = result.get("retrieval_history", [])
        if retrieval_history:
            print(f"\n📚 知识库检索记录 ({len(retrieval_history)} 轮):")
            for i, entry in enumerate(retrieval_history, 1):
                grade = entry.get("grade", "?")
                print(f"   第{i}轮: \"{entry['query'][:50]}\" → {grade}")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
