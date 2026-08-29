from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chat_models import init_chat_model
from rich import print as rprint


######1、提供大模型#########
load_dotenv(override=True)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

model = init_chat_model(
    model="MiniMax-M3",
    model_provider="openai",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL
)


######2、提供提示词模板#########
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个友好的AI助手，你的名字叫{name}"),
    ("human","你好，最近怎么样？"),
    ("ai","我很好，谢谢"),
    ("human","{user_input}")
])

# 调用
prompt_value = chat_prompt_template.invoke({"name":"小智","user_input":"2 + 2 = ？"})

######3、模型调用#########
response = model.invoke(prompt_value)
rprint(response)