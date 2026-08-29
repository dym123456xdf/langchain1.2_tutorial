from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from rich import print as rprint
import os
from dotenv import load_dotenv

# 加载配置文件
load_dotenv(override=True)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

# 获取大模型
model = init_chat_model(
    model="MiniMax-Text-01",
    model_provider="openai",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)

response = model.invoke([HumanMessage(content="2 + 3 * 2 = ?")])
print(type(response))
rprint(response)
