from dataclasses import dataclass

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

from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage


class ContactInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户邮箱地址")
    phone: str = Field(description="用户的手机号")

class EventInfo(BaseModel):
    """事件详情"""
    event_name: str = Field(description="事件名称")
    date: str = Field(description="事件发生日期")

agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=Union[ContactInfo, EventInfo])
)

response = agent.invoke(
    {
        "messages": [
            HumanMessage("从这段话中抽取结构化信息：小明的邮箱地址为：dym@atguigu.com，手机号：12345678912")
        ]
    }
)

for msg in response["messages"]:
    msg.pretty_print()

# print(response["structured_response"])

response = agent.invoke(
    {
        "messages": [
            HumanMessage("从这段话中抽取结构化信息：2026年高考报名人数突破1200万")
        ]
    }
)

for msg in response["messages"]:
    msg.pretty_print()

print(response["structured_response"])