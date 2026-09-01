from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)

AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL")

model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL
)

from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from rich import print as rprint

class ContactInfo(TypedDict):
    """用户的联系方式"""
    name: Annotated[str, ..., "用户姓名"]
    email: Annotated[str, ..., "用户邮箱地址"]
    phone: Annotated[str, ..., "用户的手机号"]

agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=ContactInfo)
)

response = agent.invoke({
    "messages" : [
        HumanMessage(content="从这段话中抽取结构化信息：小明的邮箱是dym@atguigu.com,电话是：13012341234")
    ]
})


rprint(response)