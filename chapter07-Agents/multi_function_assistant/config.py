"""
配置模块：加载环境变量 + 初始化 chat model

职责单一：仅负责 model 的初始化，不做任何业务逻辑。
"""
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从 .env 文件中加载环境变量
load_dotenv(override=True)

AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL")

model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL
)
