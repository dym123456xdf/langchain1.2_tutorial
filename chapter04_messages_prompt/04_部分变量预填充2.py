from langchain_core.prompts import ChatPromptTemplate

# 原始模板
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，目标用户是{audience}"),
    ("user", "{task}")
])

# 部分变量预填充
final_template = template.partial(role="导游",audience="游客")


result1 = final_template.invoke({"task":"介绍一下北京的故宫"})
result2 = final_template.invoke({"task":"介绍一下北京的颐和园"})

print(result1)
print(result2)