from langchain_text_splitters import MarkdownTextSplitter

markdown_text = """
# 一级标题\n
这是一级标题下的内容\n\n
## 二级标题\n
- 二级下列表项1\n
- 二级下列表项2\n
"""

# 关键步骤：直接修改实例属性
splitter = MarkdownTextSplitter(chunk_size=30, chunk_overlap=0)
splitter._is_separator_regex = True  #  强制将分隔符视为正则表达式

# 执行分割
docs = splitter.create_documents(texts = [markdown_text])

# print(len(docs))

for i, doc in enumerate(docs):
    print(f"\n🔍 分块 {i + 1}:")
    print(doc.page_content)