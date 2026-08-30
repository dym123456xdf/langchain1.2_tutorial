"""
@Author:dym
@Desc: bind_tools 最小示例 —— 用 MiniMax 模型决定是否调用 get_weather
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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

# 3、初始化模型 —— MiniMax 通过 OpenAI 兼容协议接入, 复用 ChatOpenAI 即可
model = ChatOpenAI(
    model="MiniMax-Text-01",     # MiniMax-CN 站主推的 chat 模型
    api_key=API_KEY,
    base_url=BASE_URL,
)


# 4、声明一个工具函数(没加 @tool 装饰器, 用 bind_tools 一样能识别参数 schema)
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "晴天, 温度 15°C",
        "上海": "多云, 温度 20°C",
        "广州": "小雨, 温度 25°C",
    }
    return f"{city}：{weather_data.get(city, '暂无数据')}"


# 5、把工具绑定到模型上
model_with_tools = model.bind_tools([get_weather])

# 6、调用模型 —— 模型自主决定是否调用工具
response = model_with_tools.invoke("北京的天气怎么样")
rprint(response)