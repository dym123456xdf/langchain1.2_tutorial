from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# 1. 基础配置
MAX_PAIRS_HISTORY = 4
EXIT_WORD = "quit"

# 2. 初始化模型
model = init_chat_model(
    model="MiniMax-M3",
    model_provider="openai",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL")
)


def keep_recent_messages(messages, max_pairs=3):
    """
    保留最近的N轮对话
    max_pairs : 保留对话的轮数 （每轮 = user + assistant）
    """

    # 分离system 消息和对话消息
    system_messages = [m for m in messages if m.get("role") == "system"]
    conversation_messages = [m for m in messages if m.get("role") != "system"]

    # 只保留最近的消息对
    recent_messages = conversation_messages[-(max_pairs * 2):]

    # 返回系统消息和最近的消息对
    return system_messages + recent_messages


# 3. 维护一个消息列表
messages = [
    {
        "role": "system",
        "content": "你是小谷姐姐，是尚硅谷教育的数字员工，也是一名耐心、友好的AI助手，可以回答学的问题"
    }
]

print(f"请输入具体的问题，当输入{EXIT_WORD}的时候，结束对话")

i = 1  # 描述对话的轮数
while True:
    print("\n", "=" * 10, f"第{i}轮对话开始", "=" * 10, "\n")

    user_input = input("请输入：")

    # 判断是否结束当前会话
    if user_input == EXIT_WORD:
        print("会话已结束，欢迎下次再来")
        break

    # 4. 将用户的信息添加到消息列表中
    messages.append({
        "role": "user",
        "content": user_input
    })

    # 5. 拼接AI回复的消息信息
    reply_content = ""

    # 6. 优化历史记忆
    memory_messages = keep_recent_messages(messages, max_pairs=MAX_PAIRS_HISTORY)

    # 7. 格式化打印当前 messages（system + 历史 user/assistant）
    print("\n", "-"*10, "当前 messages", "-"*10)
    for m in memory_messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        print(f"[{role}]\n{content}\n")
    print("-"*30)
    messages = memory_messages

    # 7. 流式输出模型的响应
    for chunk in model.stream(memory_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply_content += chunk.content

    print("\n", "=" * 10, f"第{i}轮对话结束", "=" * 10, "\n")

    i += 1

    # 8. 将模型的响应添加到消息列表
    messages.append({"role": "assistant", "content": reply_content})


