"""
@Author:dym
@Desc: 多工具的循环调用 —— 把多个 @tool 同时绑给模型,
       模型可以一次决定调多个, 也可以连续多轮调, 我们用 while 循环一直跑到模型不再调为止.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from rich import print as rprint


# ============================================================
# 1、读取 .env 配置(.env 里的变量优先级最高, override=True 会覆盖同名 shell 环境变量)
# ============================================================
load_dotenv(override=True)

# ============================================================
# 2、参数校验:防止 .env 缺字段时给一个晦涩的 pydantic ValidationError
# ============================================================
API_KEY  = os.getenv("MINIMAX_API_KEY")
BASE_URL = os.getenv("MINIMAX_BASE_URL")
if not API_KEY:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_API_KEY。请在项目根目录 .env 中配置:\n"
        "    MINIMAX_API_KEY=你的密钥\n"
        "    MINIMAX_BASE_URL=https://api.minimaxi.com/v1"
    )
if not BASE_URL:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_BASE_URL。请在 .env 中配置,例如:\n"
        "    https://api.minimaxi.com/v1"
    )

# ============================================================
# 3、初始化模型 —— MiniMax 通过 OpenAI 兼容协议接入, 复用 init_chat_model 即可.
#    model_provider="openai" 是必须的 (缺了 SDK 会跑去 api.openai.com);
#    base_url 必须带尾部 /v1 (SDK 会自动拼 /chat/completions);
#    模型名必须用 MiniMax 平台上真实存在的型号, 不能拿 agent 自己的 model 名当 minimax 模型名写进来.
# ============================================================
model = init_chat_model(
    model="MiniMax-Text-01",          # MiniMax-CN 站主推的 chat 模型
    model_provider="openai",          # 走 OpenAI 兼容协议, 必须显式声明
    api_key=API_KEY,
    base_url=BASE_URL,
)

from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

@tool(parse_docstring=True)
def get_news() -> str:
    """
    获取当日新闻
    """
    return "近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火热潮”，回收价格普遍上涨，旧手机成“香饽饽”。"

model_with_tools = model.bind_tools([get_weather, get_news])

messages = [
    HumanMessage("今天杭州天气如何？今天新闻是什么？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()


messages.append(response)

for tool_call in response.tool_calls:
    if tool_call["name"] == "get_weather":
        tool_msg = get_weather.invoke(tool_call)
        print(tool_msg)
        messages.append(tool_msg)
    elif tool_call["name"] == "get_news":
        tool_msg = get_news.invoke(tool_call)
        print(tool_msg)
        messages.append(tool_msg)
    else:
        raise Exception("不存在的工具")

final_response = model.invoke(messages)
messages.append(final_response)

for msg in messages:
    msg.pretty_print()