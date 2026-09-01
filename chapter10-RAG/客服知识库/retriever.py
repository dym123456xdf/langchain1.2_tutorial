"""
职责：纯检索 —— 输入 query，返回 Milvus 原始 hits。
不打印、不格式化、不构造 prompt。
"""
from config import COLLECTION_NAME
from embed_model import embed_query


def retrieve(client, query: str, limit: int = 3):
    """
    把 query 向量化后从 Milvus 中取 top-k 命中。
    返回 Milvus search 的原始 hit 列表（含 entity.text/chunk_id/source 等）。
    """
    q_vec = embed_query(query)
    hits = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_vec],
        limit=limit,
        output_fields=["text", "chunk_id", "source"],
    )
    return hits[0]