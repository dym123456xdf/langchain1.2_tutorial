"""
职责：把 chunks / query 转成向量数组。

本文件只是对 clients.get_embed_model() 的薄封装 + 提供批量/单条接口，
避免业务方直接调用 embedder.embed_documents(...) 散落各处。
"""
from clients import get_embed_model


def embed_chunks(chunks) -> list[list[float]]:
    """批量把 chunks（Document 列表）转成向量数组，与 chunks 顺序一一对应。"""
    embed_model = get_embed_model()
    texts = [c.page_content for c in chunks]
    return embed_model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """把单条查询字符串转成向量。"""
    return get_embed_model().embed_query(str(query))