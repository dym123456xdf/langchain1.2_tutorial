from langchain_core.prompts import ChatPromptTemplate
from rich import print as rprint


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个友好的AI助手，你的名字叫{name}"),
    ("human","你好，最近怎么样？"),
    ("ai","我很好，谢谢"),
    ("human","{user_input}")
])

# 调用
result = chat_prompt_template.format(name="小智",user_input="2 + 2 = ?")
rprint(result)
rprint(type(result))