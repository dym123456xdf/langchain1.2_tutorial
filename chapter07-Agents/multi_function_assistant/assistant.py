"""
SmartAssistant：组合 model + tools + prompt，创建 langchain agent。
"""
from langchain.agents import create_agent

from .config import model
from .tools import ALL_TOOLS
from .prompts import SYSTEM_PROMPT


class SmartAssistant:
    """多功能智能助手"""
    def __init__(self):
        # 初始化模型
        self.model = model

        # 工具列表
        self.tools = ALL_TOOLS

        # 系统提示词
        system_prompt = SYSTEM_PROMPT

        # ✅ 创建 agent
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=system_prompt
        )

        # 对话历史
        self.messages = []

    def chat(self, user_input: str) -> str:
        """对话接口"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 调用 agent
        result = self.agent.invoke({"messages": self.messages})

        # 更新消息历史
        self.messages = result["messages"]

        # 返回最后一条 AI 消息
        for msg in reversed(self.messages):
            if msg.type == "ai" and msg.content:
                return msg.content

        return "抱歉，我无法处理这个请求。"

    def reset(self):
        """重置对话历史"""
        self.messages = []
