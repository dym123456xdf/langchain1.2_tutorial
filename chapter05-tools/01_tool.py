"""
@Author:dym
@Desc: @tool 装饰器最小示例 —— 定义一个工具函数并直接调用
"""
from langchain_core.tools import tool
from rich import print as rprint


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    # 实际项目里这里会调天气 API，这里用本地字典模拟，便于脱离网络运行
    weather_data = {
        "北京": "晴天，温度 15°C",
        "上海": "多云，温度 20°C",
        "广州": "小雨，温度 25°C",
    }
    return f"{city}：{weather_data.get(city, '暂无数据')}"


# 1、查看工具元信息（@tool 装饰后函数变成了 BaseTool 对象）
rprint("工具名称:", get_weather.name)
rprint("工具描述:", get_weather.description)
rprint("参数 schema:", get_weather.args)
rprint("=" * 60)

# 2、直接调用工具（用 .invoke 传入参数字典）
result = get_weather.invoke({"city": "北京"})
rprint("调用结果:", result)
