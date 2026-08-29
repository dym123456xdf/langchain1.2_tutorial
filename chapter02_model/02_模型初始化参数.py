from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

# 加载配置文件
load_dotenv(override=True)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")

# 缺一不可：缺失时直接报错并提示，避免把 None 透传到 SDK 触发不友好的 ValidationError
if not MINIMAX_API_KEY:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_API_KEY。请在项目根目录新建 .env 文件并写入：\n"
        "    MINIMAX_API_KEY=你的密钥\n"
        "    MINIMAX_BASE_URL=https://api.minimaxi.com/v1"
    )
if not MINIMAX_BASE_URL:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_BASE_URL。请在 .env 中配置，例如：\n"
        "    https://api.minimaxi.com/v1"
    )

# 获取大模型
# minimax 通过 OpenAI 兼容协议接入，init_chat_model 的 provider 写 "openai"，
# 再用 base_url 指向 minimax 即可（中国站 api.minimaxi.com/v1，海外站 api.minimax.chat/v1）。
# 模型名用 MiniMax 平台上真实存在的型号（MiniMax-Text-01 / abab6.5s-chat 等），
# 不要拿 agent 自己的 model 名当 minimax 模型名写进来。
model = init_chat_model(
    model="MiniMax-Text-01",
    model_provider="openai",
    temperature=0,
    api_key=MINIMAX_API_KEY,
    base_url=MINIMAX_BASE_URL,
)

for i in range(3):
    print(f"===== 第 {i + 1} 次 =====")
    print(model.invoke("帮我写一首关于春天的七言绝句").content)
    print("=" * 60)

