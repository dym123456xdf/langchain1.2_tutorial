# 1.导入相关依赖
from langchain_text_splitters import CharacterTextSplitter
import tiktoken  # 用于计算Token数量


# 2.定义通过Token切割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", # 使用 OpenAI 的编码器
    chunk_size=18,
    chunk_overlap=0,
    separator="。",  # 指定中文句号为分隔符
    keep_separator=False,  # chunk中是否保留分隔符
)
# 3.定义文本
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。今天天气很好，想出去踏青。但是又比较懒不想出去，怎么办"

# 4.开始切割
texts = text_splitter.split_text(text)

print(f"原始文本被分割成了 {len(texts)} 个块:")
for i, chunk in enumerate(texts):
    print(f"块 {i+1}: 长度：{len(chunk)} 内容：{chunk}")
    print("-" * 50)