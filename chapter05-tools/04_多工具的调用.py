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


# ============================================================
# 4、定义工具 —— 用 docstring (parse_docstring=True) 提供参数说明.
#    注意两个细节:
#      a) 必填参数 city/company 都要写成位置参数 (无 default),
#         否则 minimax 看到的 schema 里没 required 字段,
#      b) docstring 的 Args: 段必须写 (type): 格式, 描述以句号结尾,
#         否则 parse_docstring 解析不出 description.
# ============================================================
@tool(parse_docstring=True)
def get_stock_price(company: str, timeframe: str = "today") -> str:
    """获取指定公司的股票价格信息.

    Args:
        company (str): 公司名称, 如: 苹果公司, 微软公司, 谷歌公司.
        timeframe (str): 时间范围, today-今日, week-本周, month-本月. 默认 today.

    Returns:
        str: 股票价格描述.
    """
    # 模拟股票数据
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 180.75},
        "微软公司": {"today": 415.86, "week": 412.30, "month": 405.42},
        "谷歌公司": {"today": 15.42,  "week": 15.20,  "month": 14.85},
    }
    if company in mock_data:
        price = mock_data[company].get(timeframe, "未知时间范围")
        return f"{company} {timeframe}价格: {price}美元"
    return f"未找到股票代码 {company} 的数据"


@tool(parse_docstring=True)
def search_news(company: str) -> str:
    """搜索指定公司的财经新闻.

    Args:
        company (str): 公司名称, 必填.

    Returns:
        str: 公司的财经新闻, 每个新闻占一行.
    """
    # 模拟新闻数据
    mock_news = {
        "苹果公司": [
            "苹果发布新款iPhone，股价上涨3%",
            "苹果与欧盟达成反垄断和解协议",
            "苹果将在印度扩大生产规模",
        ],
        "微软公司": [
            "微软Azure云业务季度增长超预期",
            "微软完成对Nuance的收购",
            "微软推出新一代AI助手Copilot",
        ],
        "谷歌公司": [
            "谷歌发布新AI模型，性能提升20%",
            "谷歌与OpenAI合作，开发新的AI助手",
            "谷歌在欧洲展开AI研究项目",
        ],
    }
    return "\n".join(mock_news.get(company, [f"未找到{company}的相关新闻"]))


# ============================================================
# 5、把多个工具同时绑到模型上 —— 这是"多工具"的核心:
#    模型可以根据问题决定调一个、调多个、或先调一个再根据结果调另一个.
# ============================================================
tools = [get_stock_price, search_news]
model_with_tools = model.bind_tools(tools)

# 用字典做 dispatch, 比 if-elif 字符串硬编码可扩展
# 加新工具时只往 tools 列表里追加, TOOL_REGISTRY 一行更新
TOOL_REGISTRY = {t.name: t for t in tools}


# ============================================================
# 6、维护消息列表 —— 每个问题用 HumanMessage 起头.
#    注释里给出几个典型的测试 prompt, 切换注释即可触发不同场景:
#       - 期望并行调两个工具
#       - 期望先调一个, 再基于结果调另一个 (循环多轮)
#       - 期望模型判定与工具无关, 直接回答 (触发 break)
#       - 期望模型对工具里没数据的公司直接回答
# ============================================================
message_list = [
    HumanMessage("苹果公司今天的股价是多少？最近有什么新闻？"),
    # HumanMessage("比较一下微软和苹果的股价"),       # 期望同时调两个 get_stock_price
    # HumanMessage("腾讯最近有什么重大新闻？"),          # 期望模型直接回答 (字典里没腾讯)
    # HumanMessage("海水为什么是咸的？"),                  # 与工具无关, 期望直接回答
]


# ============================================================
# 7、循环调用 —— 模型可能一轮里调多个, 也可能调完一轮再调下一轮,
#    所以用 while 而不是 for, 直到模型说"我不调了"再 break.
#
#    关键守卫 (沿用 04_args_schema/04_docstring 同款修复):
#      response.tool_calls=[] 时立刻 break, 不要再追 ToolMessage,
#      也不要再调一次 model.invoke, 否则 messages 序列错位,
#      minimax 会报 2013 'tool call and result not match'.
# ============================================================
rprint("[bold cyan]开始多工具循环调用:[/bold cyan]\n")

while True:
    response = model_with_tools.invoke(message_list)
    message_list.append(response)

    if not response.tool_calls:
        rprint("[bold yellow]模型决定直接回答, 退出循环.[/bold yellow]\n")
        break

    rprint(f"[bold cyan]模型决定调用 {len(response.tool_calls)} 个工具:[/bold cyan]")
    for tool_call in response.tool_calls:
        rprint(f"  - {tool_call['name']}({tool_call['args']})")

        tool_fn = TOOL_REGISTRY[tool_call["name"]]
        tool_message = tool_fn.invoke(tool_call)
        message_list.append(tool_message)


# ============================================================
# 8、回放整条消息链 —— 看 user / ai(tool_calls) / tool / ai ... 的交替
# ============================================================
rprint("\n[bold cyan]完整消息链:[/bold cyan]")
for msg in message_list:
    msg.pretty_print()
