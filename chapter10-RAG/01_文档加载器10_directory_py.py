# 1.导入相关的依赖
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PythonLoader
from pprint import pprint

# 2.定义DirectoryLoader对象,指定要加载的文件夹路径、要加载的文件类型和是否使用多线程
directory_loader = DirectoryLoader(
    path="../asset/load",
    glob="*.py", # 文件匹配模式（过滤器）。使用标准的 Unix 路径通配符。
    use_multithreading=True, # 是否启用多线程。填 True 意味着 LangChain 会同时并发读取多个文件。
    show_progress=True, # 是否显示进度条。填 True 时，控制台在加载文件时会弹出一个进度条
    loader_cls=PythonLoader # 指定底层核心加载器
)

# 3.加载
docs = directory_loader.load()

# 4.打印
print(len(docs))
for doc in docs:
    pprint(doc)