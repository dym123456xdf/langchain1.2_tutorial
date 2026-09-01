"""
天气查询工具
"""
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气信息

    支持中国主要城市的天气查询

    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"等

    Returns:
        包含温度、天气状况、空气质量的详细信息

    Examples:
        get_weather("北京") 返回 "多云，15-22℃，空气质量良"
    """
    weather_db = {
        "北京": "多云，15-22℃，空气质量良，湿度 45%",
        "上海": "晴天，18-25℃，空气质量优，湿度 60%",
        "深圳": "小雨，22-28℃，空气质量优，湿度 75%",
        "成都": "阴天，16-23℃，空气质量良，湿度 70%",
        "杭州": "晴天，17-24℃，空气质量优，湿度 55%",
        "广州": "多云，21-29℃，空气质量良，湿度 72%"
    }

    result = weather_db.get(city)
    if result:
        return f"{city}：{result}"
    else:
        return f"抱歉，暂不支持查询{city}的天气信息。当前支持：北京、上海、深圳、成都、杭州、广州"
