"""
@Author:dym
@Desc: tool_choice 的四种取值 —— 控制模型"调工具"的自由度.

OpenAI 协议下 bind_tools(..., tool_choice=...) 支持四种值:
  - "auto"        : 模型自己决定调不调 (默认行为, 不传 tool_choice 等价于此)
  - "none"        : 禁止调工具 —— 即使有工具, 模型也必须直接回答
  - "any"         : 强制调工具 —— 模型必须从提供的工具里挑一个调,
                     但不能调"非工具"的文字回复 (也常写作 "required")
  - {"type":"function","function":{"name":"<tool_name>"}}:
                     强制调指定工具 —— 不允许模型挑别的

注意 tool_choice="any" 和指定工具名两种用法下, 模型一定会返回 tool_calls,
必须接 ToolMessage 后再调第二轮, 否则 messages 序列错位会触发
minimax 2013 'tool call and result not match'.
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


# ============================================================
# 4、定义工具 —— 用 docstring (parse_docstring=True) 提供参数说明.
#    Args 段必须写 (type): 格式 + 描述以句号结尾, 否则 parse_docstring 解析不出 description.
# ============================================================
@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """获取当日天气.

    Args:
        city (str): 城市名称, 必填.

    Returns:
        str: 天气描述.
    """
    return f"{city}当天晴朗"


# 用户问题 —— 用同一个 query 串四个 tool_choice, 直观对比模型行为差异
QUERY = "北京今天的天气如何？"
TOOL_REGISTRY = {get_weather.name: get_weather}


def run_round(label: str, model_with_tools, question: str) -> HumanMessage | None:
    """跑一轮完整调用 + (必要时) 工具执行 + 第二轮模型调用, 返回最终的 AIMessage.

    用法: 想看 tool_choice="none" 时直接返回 final_response.content 即可;
          想看 tool_choice="any" 时 final_response 是基于工具结果的回复.
    """
    rprint(f"\n[bold cyan]{label}[/bold cyan]")

    messages = [HumanMessage(question)]

    # 第一轮: 模型决定调不调工具
    response = model_with_tools.invoke(messages)
    messages.append(response)

    # 第二轮 (只有调了工具时才跑): 把工具结果喂回去
    if response.tool_calls:
        rprint(f"  模型决定调用: {[tc['name'] for tc in response.tool_calls]}")
        for tool_call in response.tool_calls:
            tool_fn = TOOL_REGISTRY[tool_call["name"]]
            tool_message = tool_fn.invoke(tool_call)
            messages.append(tool_message)

        final_response = model.invoke(messages)
        messages.append(final_response)
    else:
        final_response = response
        rprint("  模型直接回答, 没有调工具.")

    rprint(f"  最终回复: {final_response.content}")
    return final_response


# ============================================================
# 5、四种 tool_choice 演示 —— 切换注释即可分别跑
# ============================================================

# (1) tool_choice="auto" —— 默认行为, 模型自己决定
#     期望: 大概率调 get_weather (因为 query 明确涉及天气)
rprint("[bold green]=== Case 1: tool_choice='auto' (默认行为) ===[/bold green]")
model_auto = model.bind_tools([get_weather])
run_round("auto: 模型自主决定", model_auto, QUERY)


# (2) tool_choice="none" —— 强制模型不调工具, 必须直接回答
#     期望: 不调工具, 即便 query 涉及天气, 模型也要硬答
rprint("\n[bold green]=== Case 2: tool_choice='none' (强制不调工具) ===[/bold green]")
model_none = model.bind_tools([get_weather], tool_choice="none")
run_round("none: 禁止调工具", model_none, QUERY)


# (3) tool_choice="any" —— 强制模型必须调工具
#     期望: 100% 调 get_weather(city="北京")
rprint("\n[bold green]=== Case 3: tool_choice='any' (强制调任意工具) ===[/bold green]")
model_any = model.bind_tools([get_weather], tool_choice="any")
run_round("any: 必须调工具", model_any, QUERY)


# (4) 指定具体工具名 —— 强制调某个特定工具
#     期望: 100% 调 get_weather, 没有别的选择
rprint("\n[bold green]=== Case 4: tool_choice={'type':'function', ...} (强制调指定工具) ===[/bold green]")
model_forced = model.bind_tools(
    [get_weather],
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)
run_round("forced: 强制调 get_weather", model_forced, QUERY)
