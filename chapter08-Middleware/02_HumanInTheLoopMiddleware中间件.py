from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 以init_chat_model为例
model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=os.getenv("AGNES_API_KEY"),
    base_url=os.getenv("AGNES_BASE_URL"),
    profile={"max_input_tokens": 128_000},
)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain.tools import tool
from rich import print as rprint


@tool
def get_weather(city: str, is_forcast: bool = False) -> str:
    """
    查询指定城市天气

    Args:
        city: 城市名称
        is_forcast: 是否包含明日天气预报？
    """
    res = f"{city}今天天气不错"
    if is_forcast:
        res += "\n明天下雨"
    return res


@tool
def get_news() -> str:
    """
    查询当日新闻
    """
    return "中方三艘油轮通过霍尔木兹海峡"


@tool
def read_email_tool(email_id: str) -> str:
    """通过邮件ID读取内容的伪函数"""
    return f"邮件ID：{email_id}\n是空的"


@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """发送邮件伪函数"""
    print(">>> 真的执行发送邮件工具了")
    return f"发送给 {recipient} 的邮件标题是：{subject}，内容：{body}"


agent = create_agent(
    model=model,
    tools=[get_weather, get_news, read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,
                "get_news": True,
                "read_email_tool": False,
                "send_email_tool": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "发送邮件中断了..."
                },
            },
            description_prefix="中断啦！！"
        ),
    ]
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke({
    "messages": [HumanMessage(content="请帮我查询今天北京的天气"
                                      "查询今日新闻"
                                      "查看ID为 'sk2131421' 的邮件内容，"
                                      "向15641685664@qq.com发送邮件，标题是'哈哈哈'，内容是：'你好啊'"
                                      "同时做这四件事")]
},
    config=config
)


rprint(response)

