from langchain_community.document_loaders import UnstructuredWordDocumentLoader

loader = UnstructuredWordDocumentLoader(
    # 文件路径
    file_path="../asset/load/05-sgg_chat.docx",
    # 加载模式:
    #   single 返回单个Document对象
    #   elements 按标题等元素切分文档
    mode="single",
)

docs = loader.load()

print(len(docs))
print(docs)
