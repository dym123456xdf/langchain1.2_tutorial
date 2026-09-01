# 1.导入相关依赖
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2.定义PyPDFLoader加载器
loader = PyPDFLoader("../asset/load/04-load.pdf")

# 3.加载和切割文档对象
docs = loader.load()   # 返回Document对象构成的list
# print(f"第0页：\n{docs[0]}")

# 4.定义切割器
text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size=200,
    chunk_size=120,
    chunk_overlap=0,
    # chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

# 5.对pdf内容进行切割得到文档对象
paragraphs = text_splitter.split_documents(docs)

for para in paragraphs:
    print(para)
    print('-------')