"""
多功能智能助手（LangChain 1.2）

按职责拆分的模块化项目结构：

    multi_function_assistant/
    ├── config.py        # .env 加载 + model 初始化
    ├── prompts.py       # 系统提示词
    ├── assistant.py     # SmartAssistant 类
    ├── main.py          # 入口（demo + 交互模式）
    └── tools/           # 工具子模块
        ├── __init__.py  # 聚合 ALL_TOOLS
        ├── weather.py     # get_weather
        ├── calculator.py  # calculator
        ├── time_info.py   # get_time_info
        ├── currency.py    # convert_currency
        └── search.py      # search_info

运行：
    cd chapter07-Agents/multi_function_assistant
    python main.py
"""
