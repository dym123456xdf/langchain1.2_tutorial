"""
@Author:dym
@Desc: 给 @tool 显式指定 args_schema (Pydantic BaseModel),
       看 convert_to_openai_tool 把它转成的 JSON Schema 长什么样,
       再把它绑到模型上跑一次完整的 tool_call 循环.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
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
# 4、用 Pydantic BaseModel 显式声明 args_schema
#    默认情况下 @tool 会从函数签名+docstring 推断参数,
#    但当参数需要 default / description / 复杂校验时,
#    显式给一个 BaseModel 更可控.
# ============================================================
class WeatherSchema(BaseModel):
    """查询天气时使用的参数结构 —— 传给 @tool 的 args_schema

    注意:city 必须用 Field(...) 显式标记为必填,
    否则 convert_to_openai_tool 转出的 JSON Schema 没有 required 字段,
    模型看到 city 有 default 就不一定会传, 进而偶发决定不调工具.
    if_forecast 保留 default=False, 作为可选参数更合理.
    """
    city: str         = Field(...,                       description="具体的城市名称")
    if_forecast: bool = Field(default=False,             description="是否包含明日天气")


@tool("get_weather_and_forecast",
      description="查询当日的天气，可以包含明天的天气预报",
      args_schema=WeatherSchema)
def get_weather(city: str, if_forecast: bool) -> str:
    """根据 city 返回当日天气, if_forecast=True 时附带明日预报"""
    res = f"{city}今天天气不错"
    if if_forecast:
        res += "\n明天下雨"
    return res


# ============================================================
# 5、把 @tool 转成 OpenAI 协议要求的 JSON Schema
#    这份 schema 就是 bind_tools 时塞进 system prompt 的那一段,
#    模型看到它才知道该不该调、调哪个、参数怎么填.
# ============================================================
tool_schema = convert_to_openai_tool(get_weather)

rprint("[bold cyan]Step 1 —— convert_to_openai_tool 输出:[/bold cyan]")
rprint(tool_schema)


# ============================================================
# 6、把工具绑定到模型上, 跑一次完整的 tool_call 循环:
#    user → model (决定调工具) → 主动执行工具 → model (基于结果回复)
#    维护 messages 列表是关键, 每一步都要把新消息 append 进去,
#    这样第二轮 invoke 时模型能看到完整上下文.
# ============================================================
# 1、将工具绑定到模型上
model_with_tools = model.bind_tools([get_weather])

# 2、维护一个消息列表
messages = [HumanMessage("今天杭州的天气怎么样？明天呢？")]

# 3、调用模型,得到响应：AIMessage
response = model_with_tools.invoke(messages)
messages.append(response)

# 4、获取响应中的tool_calls字段信息
# 关键守卫:如果模型第一轮没决定调工具 (tool_calls=[]),
# 就不要再 append ToolMessage 也不要跑第二轮 invoke,
# 否则 messages 序列错位, 第二轮会被 minimax 报 2013 'tool call and result not match'.
tool_calls = response.tool_calls

if tool_calls:
    for tool_call in tool_calls:
        # 5、调用工具（因为大模型不能直接调用工具，所以此时我们主动让工具调用执行）
        # 调用完，返回ToolMessage的实例
        tool_message = get_weather.invoke(tool_call)
        messages.append(tool_message)

    # 6、调用模型,得到AIMessage
    final_response = model.invoke(messages)

    # 7、添加到消息列表中
    messages.append(final_response)

# 8、遍历消息列表
for msg in messages:
    msg.pretty_print()

