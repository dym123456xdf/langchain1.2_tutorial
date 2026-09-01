# 1.导入相关依赖
from langchain_text_splitters import CharacterTextSplitter

# 2.定义要分割的文本
text = "这是第一段文本。这是第二段内容。最后一段结束。"

# 3.定义字符分割器
text_splitter = CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=8,
    separator="。",
    # keep_separator=True #chunk中是否保留切割符
)

# 4.分割文本
chunks = text_splitter.split_text(text)

# 5.打印结果
for  i,chunk in enumerate(chunks):
    print(f"块 {i + 1}:长度：{len(chunk)}")
    print(chunk)
    print("-"*50)