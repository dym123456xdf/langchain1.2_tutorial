# 1.导入相关依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2.打开.txt文件
with open("../asset/load/09-ai.txt", encoding="utf-8") as f:
    state_of_the_union = f.read()  #返回的是字符串

# 3.定义RecursiveCharacterTextSplitter（递归字符分割器）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    #chunk_overlap=0,
    length_function=len
)

# 4.分割文本
texts = text_splitter.create_documents([state_of_the_union])

# 5.打印分割文本
for text in texts:
    print(f"🔥{text.page_content}")