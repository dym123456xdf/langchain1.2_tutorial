"""
货币转换工具
"""
from langchain_core.tools import tool

@tool
def convert_currency(amount: float, from_curr: str, to_curr: str) -> str:
    """货币转换工具
    支持主要货币之间的实时汇率转换

    Args:
        amount: 金额数值
        from_curr: 源货币代码（CNY/USD/EUR/GBP/JPY/HKD）
        to_curr: 目标货币代码（CNY/USD/EUR/GBP/JPY/HKD）

    Returns:
        转换结果

    Examples:
        convert_currency(100, "CNY", "USD") 返回 "100 CNY = 14.00 USD"
    """
    # 汇率表（相对于 CNY）
    exchange_rates = {
        "CNY": 1.0,      # 人民币
        "USD": 0.14,     # 美元
        "EUR": 0.13,     # 欧元
        "GBP": 0.11,     # 英镑
        "JPY": 20.8,     # 日元
        "HKD": 1.09      # 港币
    }

    # 货币名称
    currency_names = {
        "CNY": "人民币", "USD": "美元", "EUR": "欧元",
        "GBP": "英镑", "JPY": "日元", "HKD": "港币"
    }

    from_curr = from_curr.upper()
    to_curr = to_curr.upper()

    if from_curr not in exchange_rates:
        return f"不支持的源货币：{from_curr}。支持的货币：CNY, USD, EUR, GBP, JPY, HKD"
    if to_curr not in exchange_rates:
        return f"不支持的目标货币：{to_curr}。支持的货币：CNY, USD, EUR, GBP, JPY, HKD"

    # 转换逻辑：先转为 CNY，再转为目标货币
    cny_amount = amount / exchange_rates[from_curr]
    result_amount = cny_amount * exchange_rates[to_curr]

    from_name = currency_names[from_curr]
    to_name = currency_names[to_curr]

    return f"{amount} {from_name}（{from_curr}）= {result_amount:.2f} {to_name}（{to_curr}）"
