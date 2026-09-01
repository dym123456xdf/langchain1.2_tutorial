from pathlib import Path

from langchain_core.documents import Document

# ==============================
# 1. 读文件（标准库 utf-8）
# ==============================
file_path = Path("../asset/load/01-langchain-utf-8.txt")
text = file_path.read_text(encoding="utf-8")

# ==============================
# 2. 装成 Document 对象
#    page_content: 文本内容
#    metadata:    来源信息（路径、大小等）
# ==============================
docs = [
    Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
        },
    )
]

print(docs)
print("=" * 60)
print(f"文档数量: {len(docs)}")
print(f"page_content 前 100 字: {docs[0].page_content[:100]}")
print(f"metadata: {docs[0].metadata}")