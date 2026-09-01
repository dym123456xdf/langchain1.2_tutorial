"""
职责：资源工厂 —— 提供 MilvusClient 与 Embeddings 单例。
不参与任何业务流程，调用方按需 get。

低耦合：使用懒加载模块级缓存；其他模块不直接 new 客户端，统一走这里。
"""
from langchain_community.embeddings import MiniMaxEmbeddings
from pymilvus import MilvusClient
from dotenv import load_dotenv
import os

from config import EMBED_MODEL_NAME, MILVUS_URI


# ---------- 懒加载单例（线程不安全，但本教程脚本是单线程，足够） ----------
_milvus_client: MilvusClient | None = None
_embed_model = None


def get_milvus_client() -> MilvusClient:
    """返回已连接到 Milvus URI 的客户端实例（单例）。"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(MILVUS_URI)
    return _milvus_client


def get_embed_model():
    """
    返回 minimax embo-01 嵌入模型实例（单例）。

    minimax embeddings 不是 OpenAI 兼容协议：
        字段为 texts + type，必须走 langchain_community.MiniMaxEmbeddings。
        init_embeddings("openai:embo-01") 服务端会返回 2013 错误。

    需要 env：
        MINIMAX_API_KEY
        MINIMAX_BASE_URL
        MINIMAX_GROUP_ID   强校验，不能为空
    """
    global _embed_model
    if _embed_model is None:
        load_dotenv(override=True)
        _embed_model = MiniMaxEmbeddings(
            model=EMBED_MODEL_NAME,
            minimax_api_key=os.getenv("MINIMAX_API_KEY"),
            minimax_group_id=os.getenv("MINIMAX_GROUP_ID"),
            endpoint_url=os.getenv("MINIMAX_BASE_URL"),  # 类默认 https://api.minimax.chat/v1/embeddings
        )
    return _embed_model