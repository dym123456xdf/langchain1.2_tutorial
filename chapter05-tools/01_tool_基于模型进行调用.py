"""
@Author:dym
@Desc: 把工具绑定到 MiniMax(原 closeai) 模型上, 让模型决定是否调用工具
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from rich import print as rprint

# 1、读取 .env 配置(.env 里的变量优先级最高, override=True 会覆盖同名 shell 环境变量)
load_dotenv(override=True)
# 2、参数校验:防止 .env 缺字段时给一个晦涩的 pydantic ValidationError
API_KEY = os.getenv("MINIMAX_API_KEY")
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

# 3、初始化模型
#    MiniMax 通过 OpenAI 兼容协议接入,直接复用 ChatOpenAI 即可。
#    base_url 必须带尾部 /v1(SDK 会自动拼 /chat/completions)。
#    模型名必须用 MiniMax 平台上真实存在的型号,不能拿 agent 自己的 model 名当 minimax 模型名写进来。
model = ChatOpenAI(
    model="MiniMax-Text-01",     # MiniMax-CN 站主推的 chat 模型
    api_key=API_KEY,
    base_url=BASE_URL,
)


# 4、定义工具 —— 用 @tool 装饰器, 注解会被转成 JSON Schema 喂给模型
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 实际项目里这里会调天气 API, 这里用本地字典模拟, 便于脱离网络运行
    weather_data = {
        "北京": "晴天, 温度 15°C",
        "上海": "多云, 温度 20°C",
        "广州": "小雨, 温度 25°C",
    }
    return f"{city}：{weather_data.get(city, '暂无数据')}"


# 5、把工具绑定到模型上(model 拿到工具描述后才能决定要不要调、调哪个)
model_with_tools = model.bind_tools([get_weather])

# 6、调用模型 —— 模型可以自主决定是否调用工具
#    例 1: 明确涉及天气, 模型大概率会调 get_weather
#    例 2: 与天气无关, 模型大概率直接回答
response = model_with_tools.invoke("北京天气如何？")
# response = model_with_tools.invoke("2 + 3 = ？")

# 7、检查模型是否决定调用工具
if response.tool_calls:
    rprint("AI 决定调用工具：")
    for call in response.tool_calls:
        rprint(f"  - 工具名: {call['name']}")
        rprint(f"    参数:   {call['args']}")
else:
    rprint("AI 直接回答:", response.content)