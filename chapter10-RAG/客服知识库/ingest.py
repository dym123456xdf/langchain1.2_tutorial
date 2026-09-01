"""
职责：入库编排 —— 把"切分好的 chunks"变成 Milvus 中可检索的数据。

业务流程：
    chunks ──(embed_model)──▶ records ──(milvus client)──▶ upsert + flush

本文件只做 orchestration，不重复实现加载/切分/向量化的细节。
"""
from config import COLLECTION_NAME, KNOWLEDGE_FILE
from document_processor import load_and_split
from embed_model import embed_chunks
from milvus_setup import setup_milvus


def build_records(chunks, vectors) -> list[dict]:
    """把 (chunks, vectors) 打包成 Milvus 接受的 data 记录列表。"""
    return [
        {
            "id": i,
            "vector": vectors[i],
            "text": chunks[i].page_content,
            "source": KNOWLEDGE_FILE,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]


def upsert_records(client, records) -> dict:
    """写入 Milvus 并 flush 磁盘，返回原始 upsert 响应。"""
    res = client.upsert(collection_name=COLLECTION_NAME, data=records)
    client.flush(collection_name=COLLECTION_NAME)
    print("insert results : ", res)
    return res


def run_ingest() -> None:
    """端到端入库：setup → 切分 → 向量化 → 写库 → 打印统计。"""
    client = setup_milvus()
    chunks = load_and_split()
    vectors = embed_chunks(chunks)
    records = build_records(chunks, vectors)
    upsert_records(client, records)

    stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
    print("stats :", stats)

    count = client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0",
        output_fields=["id", "chunk_id"],
    )
    print(f"collection 中现有记录数 = {len(count)}")


if __name__ == "__main__":
    run_ingest()