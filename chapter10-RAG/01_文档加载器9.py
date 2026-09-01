# 1.导入相关的依赖
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pprint import pprint

# 2.定义UnstructuredMarkdownLoader对象
loader = UnstructuredMarkdownLoader(
    file_path="../asset/load/06-load.md",
    # 加载模式:
    #   single 返回单个Document对象
    #   elements 按标题等元素切分文档
    mode= "single",
    # 解析策略：
    #   "fast"（快速模式），它会以最快的速度提取文本，不进行复杂的版面分析
    #   "hi_res" 高分辨率模式
    strategy="fast"
)

# 3.加载
docs = loader.load()

# 4.打印
print(len(docs))
pprint(docs)