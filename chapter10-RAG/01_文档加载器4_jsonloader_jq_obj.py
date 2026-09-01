# 1.导入相关依赖
from langchain_community.document_loaders import JSONLoader
from rich import print as rprint

# 2.定义json文件的路径
file_path = '../asset/load/03-response.json'

# 3.定义JSONLoader对象
# 需求1：提取data.items中的数据
# loader = JSONLoader(
#     file_path=file_path,  # 文件路径
#     jq_schema=".data.items[]",
#     text_content=False,  # 提取内容是否为字符串格式
# )


# 需求2：提取data.items[].content中的数据
# loader = JSONLoader(
#     file_path=file_path,  # 文件路径
#     jq_schema=".data.items[].content",
# )


# 需求3：提取data.items中指定字段的数据
loader = JSONLoader(
    file_path=file_path,  # 文件路径
    jq_schema="""
        .data.items[] | {
            author,
            created_at,
            content: (.title + "\n" + .content)
        }
    """,
    text_content=False,  # 提取内容是否为字符串格式
)


# 4.加载
data = loader.load()
rprint(data)
