from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# 以init_chat_model为例
model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=os.getenv("AGNES_API_KEY"),
    base_url=os.getenv("AGNES_BASE_URL"),
    profile={"max_input_tokens": 128_000},
)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent

messages = [
    SystemMessage("你是个非常友好的AI助手"),
    HumanMessage("你好啊，我是老王，你是谁？"),
    AIMessage("你好老王，我是小王"),
    HumanMessage("好的小王，很高兴认识你"),
    AIMessage("你高兴得太早了"),
    HumanMessage("呵呵，你什么意思")
]

agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=[
                ("tokens", 100),
                ("messages", 6),
                ("fraction", 0.001)
            ],
            keep=("messages", 2),
            summary_prompt="对历史消息摘要，消息列表如下\n{messages}"

        )
    ]
)

response = agent.invoke({
    "messages": messages
})

for msg in response["messages"]:
    msg.pretty_print()
