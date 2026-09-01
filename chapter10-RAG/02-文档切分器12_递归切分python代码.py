# 1.导入相关依赖
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pprint import pprint

# 2.定义要分割的python代码片段
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

def hello_world1():
    print("Hello, World1!")
"""

# 3.定义递归字符切分器
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0
)

# 4.文档切分
python_docs = python_splitter.create_documents(texts=[PYTHON_CODE])

pprint(python_docs)