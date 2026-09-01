"""
工具聚合：从各子模块导入，组合为 ALL_TOOLS 列表。

新增加工具时，只需：
  1) 在 tools/ 下新增一个文件，定义 @tool 函数
  2) 在本文件 from .xxx import your_tool 并加入 ALL_TOOLS
"""
from .weather import get_weather
from .calculator import calculator
from .time_info import get_time_info
from .currency import convert_currency
from .search import search_info

ALL_TOOLS = [
    get_weather,
    calculator,
    get_time_info,
    convert_currency,
    search_info,
]

__all__ = ["ALL_TOOLS", "get_weather", "calculator", "get_time_info", "convert_currency", "search_info"]
