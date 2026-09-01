from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

from langgraph.checkpoint.memory import InMemorySaver

# 从.env文件中加载环境变量
load_dotenv(override=True)

AGNES_API_KEY = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL")

model = init_chat_model(
    model="agnes-2.5-flash",
    model_provider="openai",
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL
)

from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any
from rich import print as rprint

@tool
def query_customer_data(customer_id: str) -> Dict[str, Any]:
    """
    查询客户基本信息

    Args:
        customer_id: 客户ID，用于唯一标识客户

    Returns:
        包含客户基本信息的字典，如姓名、等级、加入日期等
    """
    # 模拟数据库查询
    return {"name": "张三","level": "VIP","join_date": "2023-01-15"}


@tool
def check_order_history(customer_id: str) -> Dict[str, Any]:
    """
    查询客户订单历史

    Args:
        customer_id: 客户ID，用于唯一标识客户

    Returns:
        包含客户订单历史的字典，如总订单数、总花费等
    """
    return {"total_orders": 15,"total_spent": 25800.00}


@tool
def get_current_promotions() -> Dict[str, Any]:
    """
    获取当前可用促销活动

    Returns:
        包含当前可用促销活动的字典，如活动名称、有效日期等
    """
    return {
        "promotions": ["老用户优惠", "会员专属折扣"],
        "valid_until": "2027-01-31"
    }


# 其他工具代码同上，保持不变
# ... ...
# 1. 创建内存检查点存储
checkpointer = InMemorySaver()

# 2. 创建Agent
customer_service_agent = create_agent(
    model=model,
    tools=[query_customer_data, check_order_history, get_current_promotions],
    checkpointer=checkpointer  # 启用检查点
)

# 3. 创建唯一的会话ID
config = {"configurable": {"thread_id": "session01"}}

# 4. 调用Agent
checkpoint_count = 0

# 使用checkpoints模式进行流式监控
for chunk in customer_service_agent.stream(
        {"messages": [{"role": "user","content": "查询客户ID为 CUST123456 的完整信息和可用优惠"}]},
        config=config,
        stream_mode="checkpoints"
):
    checkpoint_count += 1
    print(f"检查点 #{checkpoint_count}")
    print(chunk)
    print("-" * 50)

