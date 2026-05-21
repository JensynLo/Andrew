from .integration import RAGIntegration, initialize_rag_system


def main():
    """
    示例主函数 - 展示如何使用 RAG 集成系统
    """
    print("初始化 RAG 系统...")
    initialize_rag_system()
    # 测试查询
    user_queries = [
        "How to obtain the Terra Blade?",
        "How to make a workbench?",
        "Where can I find the Merchant NPC?",
    ]
    rag_system = RAGIntegration()
    rag_system.load_index(folder_path="output/ragindex")

    print("\n开始 RAG 查询测试...")
    for query in user_queries:
        print(f"\n用户问题: {query}")

        # 获取答案和上下文
        result = rag_system.query_with_context(query)

        print(f"模型回答: {result['answer']}")
        print(f"使用了 {len(result['retrieved_docs'])} 个相关文档作为上下文")

        # 显示检索到的文档信息
        print("检索到的相关文档:")
        for i, doc in enumerate(result["retrieved_docs"]):
            print(
                f"  - 文档 {i + 1}: 类型={doc['metadata'].get('type', 'unknown')}, 内容={doc['content'][:100]}..."
            )

    print("\n交互式模式，输入 'quit' 退出")
    while True:
        user_query = input("\n请输入您的问题: ").strip()
        if user_query.lower() in ["quit", "exit", "q"]:
            break

        if user_query.strip():
            try:
                result = rag_system.query_with_context(user_query)
                print(f"\n🤖 回答: {result['answer']}")

                # 可选：显示检索到的上下文
                show_context = input("是否显示检索到的上下文? (y/n): ").strip().lower()
                if show_context == "y":
                    print(f"\n📋 检索到的上下文:")
                    for i, doc in enumerate(result["retrieved_docs"]):
                        print(
                            f"  [{i + 1}] 类型: {doc['metadata'].get('type', 'unknown')}"
                        )
                        print(f"      内容: {doc['content'][:200]}...")
            except Exception as e:
                print(f"处理查询时出错: {e}")


if __name__ == "__main__":
    main()
