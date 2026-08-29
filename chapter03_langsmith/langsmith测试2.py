import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 将env文件中的变量加载为环境变量
# override=True：表示.env优先
load_dotenv(override=True)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

model = init_chat_model(
    model="MiniMax-M3",
    model_provider="openai",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)

print(model.invoke("你好，用一句话回答"))