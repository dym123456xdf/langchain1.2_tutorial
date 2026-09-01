"""
职责：把 Milvus 检索结果（hits）→ 格式化 context → user prompt。
纯数据变换，无 IO，无打印。
"""
def format_hits(hits) -> list[str]:
    """
    把 hits 列表转换为带编号 / 元数据的 context 块列表。
    每块形如：[片段1 | chunk_id=0 | source=../knowledge.txt]\n<正文>
    """
    blocks = []
    for i, hit in enumerate(hits, 1):
        text = hit["entity"]["text"]
        source = hit["entity"].get("source", "unknown")
        chunk_id = hit["entity"].get("chunk_id", "unknown")

        blocks.append(
            f"[片段{i} | chunk_id={chunk_id} | source={source}]\n{text}"
        )
    return blocks


def build_context(hits) -> str:
    """把多个 context 块合并成一个字符串，块与块之间空一行。"""
    return "\n\n".join(format_hits(hits))


def build_user_prompt(query: str, context: str) -> str:
    """拼出最终发给 agent 的 user prompt。"""
    return f"""问题：
{query}

上下文：
{context}
"""