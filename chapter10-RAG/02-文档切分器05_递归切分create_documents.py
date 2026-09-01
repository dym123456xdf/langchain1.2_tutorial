# 1.导入相关依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2.定义RecursiveCharacterTextSplitter分割器对象
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    add_start_index=True,
)

# 3.定义拆分的内容
text="LangChain框架特性\n\n多模型集成(GPT/Claude)\n记忆管理功能\n链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。"

# 4.拆分器分割
paragraphs = text_splitter.split_text(text)

for i,chunk in enumerate(paragraphs):
    print(f"块{i + 1},长度：{len(chunk)}")
    print(chunk)
    print('-' * 50)