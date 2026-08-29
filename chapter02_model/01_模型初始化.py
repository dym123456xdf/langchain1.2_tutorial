import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1、读取 .env 配置文件中的信息。.env 文件里的变量优先级最高
load_dotenv(override=True)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
print(MINIMAX_API_KEY)
# minimax 海外/中国站 base_url 不同：海外 api.minimax.chat，中国 api.minimaxi.com
# 这里按你刚才确认的中国站配置；如要切海外站，把 base_url 改成 https://api.minimax.chat/v1 即可
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
print(MINIMAX_BASE_URL)

# 缺一不可：缺失时直接报错并提示，避免把 None 透传到 SDK 触发不友好的 ValidationError
if not MINIMAX_API_KEY:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_API_KEY。请在项目根目录新建 .env 文件并写入：\n"
        "    MINIMAX_API_KEY=你的密钥\n"
        "    MINIMAX_BASE_URL=https://api.minimaxi.com/v1"
    )

# 2、模型初始化
# minimax 通过 OpenAI 兼容协议接入，复用 ChatOpenAI 即可。
# 默认模型选 MiniMax-M（用户当前正在用的模型），如需切换可改成 abab6.5s-chat / MiniMax-Text-01 等。
llm_minimax = ChatOpenAI(
    model="MiniMax-M3",
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)

# 3、模型调用
response = llm_minimax.invoke("请用一句话介绍你自己")
print(response)
