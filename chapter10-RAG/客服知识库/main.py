"""
职责：单次问答编排。
    query ──▶ retrieve ──▶ prompt_builder ──▶ agent ──▶ 打印回答
"""
from agent_setup import build_agent
from clients import get_milvus_client
from prompt_builder import build_context, build_user_prompt
from retriever import retrieve


def print_hits(hits) -> None:
    """打印检索命中（仅展示用，业务流程不依赖）。"""
    print("=== 检索结果 ===")
    for i, hit in enumerate(hits, 1):
        text = hit["entity"]["text"]
        source = hit["entity"].get("source", "unknown")
        chunk_id = hit["entity"].get("chunk_id", "unknown")
        score = hit["distance"]   # COSINE：score 越大越相似
        print(f"[{i}] chunk_id={chunk_id} score={score:.4f} source={source}")
        print(text)
        print()


def answer(query: str, limit: int = 5) -> None:
    """对单个 query 执行检索 → 构造 prompt → 调用 agent → 打印回答。"""
    client = get_milvus_client()

    hits = retrieve(client, query, limit=limit)
    print_hits(hits)

    context = build_context(hits)
    user_prompt = build_user_prompt(query, context)

    agent = build_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_prompt}],
    })

    print("====最终回答====")
    result["messages"][-1].pretty_print()


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    q = "为什么我在 7 天内申请退款，还是被拒了？"
    answer(q)