# 1.导入依赖
from  langchain_community.document_loaders import JSONLoader
from rich import print as rprint

# 2.定义JSONLoader对象
# 情况1
# json_loader=JSONLoader(
#     file_path="../asset/load/03-load.json",
#     jq_schema=".", #直接提取完整的JSON对象（包括所有字段）
#     text_content=False #保持原始 JSON 结构，将提取的数据转换为JSON字符串存入page_content字段中
# )

# 情况2
# .messages[].content:遍历.messages[]中所有元素 从每一个元素中提取.content字段
json_loader=JSONLoader(
    file_path="../asset/load/03-load.json",
    jq_schema=".messages[].content"
)

# 3.加载
docs = json_loader.load()
rprint(docs)