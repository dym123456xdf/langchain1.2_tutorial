"""
@Author:dym
@Desc: 不使用 @tool 装饰器 —— 通过 convert_to_openai_tool 把普通函数喂给模型
"""
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
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


# 4、声明一个普通函数(注意:没加 @tool 装饰器, 也没有 docstring)
def get_weather(city: str) -> str:
    return f"{city}天气晴朗~~"


# 5、用 convert_to_openai_tool 把函数转成 OpenAI 兼容的工具描述
#    这一步是"不使用 @tool 也能让模型识别工具"的关键 —— 函数注解/字段信息会被抽成 JSON Schema
tool_schema = convert_to_openai_tool(get_weather)
rprint("convert_to_openai_tool 生成的 schema:")
rprint(json.dumps(tool_schema, ensure_ascii=False, indent=2))
rprint("=" * 60)

# 6、把工具绑定到模型上(可以直接传 schema dict, 也可以传函数本身让模型自己转换)
#    传 schema 的方式就是"绕开 @tool"的标准写法
model_with_tools = model.bind_tools([tool_schema])

# 7、调用模型 —— 模型自主决定是否调用工具
response = model_with_tools.invoke("北京的天气怎么样")
rprint(response)