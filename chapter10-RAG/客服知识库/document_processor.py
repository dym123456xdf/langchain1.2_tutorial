"""
职责：把磁盘上的文本文件 → Document chunks（langchain Document 列表）。
纯 IO + 切分，不涉及任何向量库/嵌入模型。
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_FILE, SEPARATORS


def load_documents(file_path: str = KNOWLEDGE_FILE, encoding: str = "utf-8"):
    """读取本地文本，返回 langchain Document 列表（一般是 1 个元素）。"""
    loader = TextLoader(file_path=file_path, encoding=encoding)
    return loader.load()


def split_documents(documents):
    """把 Document 列表切分成 chunks。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_documents(documents)


def load_and_split(file_path: str = KNOWLEDGE_FILE):
    """一步：加载 + 切分。返回 chunks。"""
    docs = load_documents(file_path)
    chunks = split_documents(docs)
    print(f"文档共切分为 {len(chunks)} 个 chunk")
    for i, chunk in enumerate(chunks):
        print(f"\nchunk{i} : ", chunk.page_content)
    return chunks