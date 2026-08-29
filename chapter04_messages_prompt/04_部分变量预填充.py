from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich import print as rprint


# 原始模板
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，目标用户是{audience}"),
    ("user", "{task}")
])


result1 = template.invoke({"role":"导游","audience":"游客","task":"介绍一下北京的故宫"})
result2 = template.invoke({"role":"导游","audience":"游客","task":"介绍一下北京的颐和园"})

rprint(result1)
rprint(result2)