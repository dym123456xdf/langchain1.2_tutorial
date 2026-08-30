"""
@Author:dym
@Desc: Agent 基本用法 —— Agnes 2.5 Flash 版本
      Agnes 是 OpenAI 兼容协议，可以直接用 init_chat_model 传入 provider:model 形式
      Agnes 的 API Key 通过环境变量 AGNES_API_KEY 配置（OpenAI 兼容风格）
"""
import os
import subprocess
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 1、读取 .env 配置。Agnes 的 Key 默认从 AGNES_API_KEY 读取，
#    但 init_chat_model 的 openai provider 只认 OPENAI_API_KEY，
#    所以这里显式塞一个给 ChatOpenAI 用，避免依赖环境变量命名约定
load_dotenv(override=True)

AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1")

if not AGNES_API_KEY:
    raise RuntimeError(
        "未找到环境变量 AGNES_API_KEY。请在 .env 文件中写入：\n"
        "    AGNES_API_KEY=你的密钥\n"
        "    AGNES_BASE_URL=https://apihub.agnes-ai.com/v1"
    )

# 2、初始化模型（两种写法等价，任选其一）
# 写法 A：init_chat_model 统一入口，provider:model 形式
model = init_chat_model(
    model="openai:agnes-2.5-flash",
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
)

# 写法 B：等价写法，直接用 ChatOpenAI
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(
#     model="agnes-2.5-flash",
#     api_key=AGNES_API_KEY,
#     base_url=AGNES_BASE_URL,
# )

# 3、创建 Agent（不传 tools 时就是一个纯对话 agent）
agent = create_agent(
    model=model,  # Agnes 2.5 Flash
)

print(type(agent))

# 4、渲染 LangGraph 流程图为 PNG（终端友好版：落盘 + macOS 预览打开）
#    替代 .ipynb 里的 IPython.display.Image —— 脚本环境没有 IPython
#    draw_mermaid_png() 内部调 mermaid.ink 渲染，需要联网
print("=" * 60)
print("正在渲染 LangGraph 流程图...")
png_bytes = agent.get_graph().draw_mermaid_png()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_images")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "agent_graph.png")
with open(out_path, "wb") as f:
    f.write(png_bytes)

print(f"流程图已保存：{out_path}")

if sys.platform == "darwin":
    subprocess.run(["open", out_path])
    print("已用 macOS 预览打开")
else:
    print(f"非 macOS 系统，请手动打开：{out_path}")