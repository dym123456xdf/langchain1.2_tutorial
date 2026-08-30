from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, AutoStrategy
from rich import print as rprint
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 1.模型初始化
model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=os.getenv("AGNES_API_KEY"),
    base_url=os.getenv("AGNES_BASE_URL")
)

# 2.使用Pydantic结构化方式定义
class ContractInfo(BaseModel):
    """用户的联系方式"""
    name : str = Field(description="用户的姓名")
    email : str = Field(description="用户的邮箱")
    phone : str = Field(description="用户的电话")

agent = create_agent(
    model = model,
    response_format=AutoStrategy(ContractInfo)
    # response_format=ContractInfo  # 不推荐大家使用
)


response = agent.invoke({
    "messages": [
        {"role":"user","content":"从以下信息中提取用户信息，小明的邮箱是dym@atguigu.com,电话是13012341234"}
    ]
})

rprint(response)