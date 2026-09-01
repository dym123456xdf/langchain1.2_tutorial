"""
职责：在指定 Milvus 实例上创建数据库与集合（保证幂等可重跑）。
被 ingest 入口触发；自身不调用业务模块。
"""
from config import COLLECTION_NAME, DB_NAME, EMBED_DIM
from clients import get_milvus_client


def ensure_database(client) -> None:
    """若数据库不存在则创建，并切换到该数据库。"""
    existed = client.list_databases()
    if DB_NAME not in existed:
        client.create_database(db_name=DB_NAME)
    client.use_database(db_name=DB_NAME)


def recreate_collection(client) -> None:
    """若集合已存在则删除（避免重复入库），再以 COSINE 度量重建。"""
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBED_DIM,
        metric_type="COSINE",
    )


def setup_milvus():
    """一步到位：建数据库 + 重建集合。返回可用 client。"""
    client = get_milvus_client()
    ensure_database(client)
    recreate_collection(client)
    print(f"[milvus_setup] ready: db={DB_NAME}, collection={COLLECTION_NAME}")
    return client