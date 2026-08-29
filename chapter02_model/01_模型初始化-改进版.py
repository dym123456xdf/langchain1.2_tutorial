import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1、读取 .env 配置。.env 文件里的变量优先级最高(override=True 会覆盖同名 shell 环境变量)
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

# 3、模型初始化
#    MiniMax 通过 OpenAI 兼容协议接入,直接复用 ChatOpenAI 即可。
#    注意:base_url 必须带尾部 /v1(SDK 会自动拼 /chat/completions)。
#    模型名必须用 MiniMax 平台上真实存在的型号,不能用我们 agent 自己内部的 model 名字。
llm_minimax = ChatOpenAI(
    model="MiniMax-Text-01",       # MiniMax-CN 站主推的 chat 模型(见 minimax-chat-completions 参考)
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 4、模型调用
response = llm_minimax.invoke("请用一句话介绍你自己")
print(llm_minimax.invoke("1 + 1 = ？"))
print(response)
