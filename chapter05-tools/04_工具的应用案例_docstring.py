"""
@Author:dym
@Desc: 用 docstring (parse_docstring=True) 给 @tool 提供参数描述,
       看 convert_to_openai_tool 转出的 JSON Schema 长什么样,
       再绑到模型上跑一次完整的 tool_call 循环.

与 args_schema 版的对比:
  - args_schema 版:用 Pydantic BaseModel 显式声明字段, 适合参数多 / 校验复杂 / 需要 default 的场景;
  - docstring 版:  用 Google 风格的 docstring + parse_docstring=True,
                   适合参数少 / 想让函数本体看起来干净的场景.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.utils.function_calling import convert_to_openai_tool
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


# ============================================================
# 4、定义工具 —— 用 docstring 写参数说明, parse_docstring=True 让 SDK 解析 Args 段.
#    注意:为了让 minimax 稳定传 city, 必须把 city 设为位置参数 (无 default),
#    这样 convert_to_openai_tool 转出的 schema 里才会有 required: ['city'].
#    (args_schema 版的修复点同款, docstring 版用函数签名 default 触发同样的坑)
# ============================================================
@tool("get_weather_and_forecast", parse_docstring=True)
def get_weather(city: str, if_forecast: bool = False) -> str:
    """查询当日的天气，可以包含明天的天气预报

    Args:
        city (str): 具体的城市名称, 必填.
        if_forecast (bool): 是否包含明天的天气, 选填, 默认 False.
    """
    res = f"{city}今天天气不错"
    if if_forecast:
        res += "\n明天下雨"
    return res


# ============================================================
# 5、把 @tool 转成 OpenAI 协议要求的 JSON Schema
#    parse_docstring=True 时, docstring 里的 description 会被注入到 schema 的 description 字段.
# ============================================================
tool_schema = convert_to_openai_tool(get_weather)

rprint("[bold cyan]Step 1 —— convert_to_openai_tool 输出 (parse_docstring):[/bold cyan]")
rprint(tool_schema)


# ============================================================
# 6、把工具绑定到模型上, 跑一次完整的 tool_call 循环.
#    关键守卫:如果模型第一轮没决定调工具 (tool_calls=[]),
#    就不要再 append ToolMessage 也不要跑第二轮 invoke,
#    否则 messages 序列错位, 第二轮会被 minimax 报 2013 'tool call and result not match'.
# ============================================================
model_with_tools = model.bind_tools([get_weather])

messages = [HumanMessage("今天杭州的天气怎么样？明天呢？")]

rprint("\n[bold cyan]Step 2 —— 第一轮 invoke (期望模型决定调工具):[/bold cyan]")
response = model_with_tools.invoke(messages)
messages.append(response)

tool_calls = response.tool_calls

if tool_calls:
    for tool_call in tool_calls:
        # 7、调用工具（因为大模型不能直接调用工具，所以此时我们主动让工具调用执行）
        tool_message = get_weather.invoke(tool_call)
        messages.append(tool_message)

    rprint("\n[bold cyan]Step 3 —— 第二轮 invoke (把工具结果喂回模型):[/bold cyan]")
    final_response = model.invoke(messages)
    messages.append(final_response)
else:
    rprint("\n[bold cyan]Step 3 —— 模型决定直接回答, 跳过第二轮 invoke.[/bold cyan]")

# 8、遍历消息列表 —— 看 user / ai(tool_calls) / tool / ai 四种角色如何交替
rprint("\n[bold cyan]Step 4 —— 完整消息链:[/bold cyan]")
for msg in messages:
    msg.pretty_print()
