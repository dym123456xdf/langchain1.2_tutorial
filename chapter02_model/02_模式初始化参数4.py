from langchain.chat_models import init_chat_model
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
    # temperature=1.9,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    max_tokens=10,
)

print(model.invoke("介绍一下你自己"))
