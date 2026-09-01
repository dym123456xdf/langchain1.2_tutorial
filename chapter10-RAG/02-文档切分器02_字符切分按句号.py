# 1.导入相关依赖
from langchain_text_splitters import CharacterTextSplitter

# 2.示例文本
text = """
LangChain 是一个用于开发由语言模型驱动的应用程序的框架的。它提供了一套工具和抽象，使开发者能够更容易地构建复杂的应用程序。
"""

# 3.定义字符分割器
splitter = CharacterTextSplitter(
    chunk_size=50, # 每块大小
    chunk_overlap=5,# 块与块之间的重复字符数
    # length_function=len,
    separator=""   # 设置为空字符串时，表示禁用分隔符优先
)

# 4.分割文本
texts = splitter.split_text(text)

# 5.打印结果
for i, chunk in enumerate(texts):
    print(f"块 {i+1}:长度：{len(chunk)}")
    print(chunk)
    print("-" * 50)