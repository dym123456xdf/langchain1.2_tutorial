from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from rich import print as rprint
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

model = init_chat_model(
    model="MiniMax-Text-01",
    model_provider="openai",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
    model_kwargs={"tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of a location, the user should supply a location first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"]
                },
            }
        },
    ]}
)
# 向模型发送单条数据
response = model.invoke("你好，今天北京的天气如何")
#response = model.invoke("1 + 2 = ？")
# 打印响应
rprint(response)