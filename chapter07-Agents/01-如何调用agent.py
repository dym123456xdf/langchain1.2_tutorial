from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from rich import print as rprint

load_dotenv(override=True)

# 以init_chat_model为例
model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=os.getenv("AGNES_API_KEY"),
    base_url=os.getenv("AGNES_BASE_URL")
)



agent = create_agent(
    model=model
)

# 调用
# response= agent.invoke({
#     "messages":[
#         {"role":"user","content":"你好"}
#     ]
# })

# response= agent.invoke({
#     "messages":[
#         "你好"
#     ]
# })

response= agent.invoke({
    "messages":[
        {"role":"system","content":"你是一个精通数学的老师，擅长以通俗易懂的方式讲解数学问题"},
        {"role":"user","content":"100 + 20 * 3 = ？"}
    ]
})

rprint(response)