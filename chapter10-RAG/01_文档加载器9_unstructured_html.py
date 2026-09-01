# 1.导入相关的依赖
from pprint import pprint

from langchain_community.document_loaders import UnstructuredHTMLLoader

# 2.定义UnstructuredHTMLLoader对象
# strategy:
#   "fast" 解析加载html文件速度是比较快（但可能丢失部分结构或元数据）
#   "hi_res": (高分辨率解析) 解析精准（速度慢一些）
#   "ocr_only"  强制使用ocr提取文本，仅仅适用于图像（对HTML无效）

# mode ：one of `{'paged', 'elements', 'single'}
#    "elements"  按语义元素（标题、段落、列表、表格等）拆分成多个独立的小文档

loader = UnstructuredHTMLLoader(
    file_path="../asset/load/07-load.html",
    mode="elements",
    strategy="fast"
)

# 3.加载
docs = loader.load()

print(len(docs))  # 16

# 4.打印
for doc in docs:
    pprint(doc)