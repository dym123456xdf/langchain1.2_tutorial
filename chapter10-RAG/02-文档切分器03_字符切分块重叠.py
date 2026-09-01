# 1.导入相关依赖
from langchain_text_splitters import CharacterTextSplitter

# 2.定义要分割的文本
text = "这是一个示例文本啊。我们将使用CharacterTextSplitter将其分割成小块。分割基于字符数。"

# text = """
# LangChain 是一个用于开发由语言模型。驱动的应用程序的框架的。它提供了一套工具和抽象。使开发者能够更容易地构建复杂的应用程序。
# """

# 3.定义分割器实例
text_splitter = CharacterTextSplitter(
    chunk_size=30,   # 每个块的最大字符数
    chunk_overlap=5, # 块之间的重叠字符数
    separator="。",  # 按句号分割优先
)

# 4.开始分割
chunks = text_splitter.split_text(text)

# 5.打印效果
for  i,chunk in enumerate(chunks):
    print(f"块 {i + 1}:长度：{len(chunk)}")
    print(chunk)
    print("-"*50)
