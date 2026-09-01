"""
信息搜索工具
"""
from langchain_core.tools import tool

@tool
def search_info(keyword: str, category: str = "all") -> str:
    """搜索各类信息

    Args:
        keyword: 搜索关键词
        category: 搜索分类
            - "product": 搜索产品
            - "news": 搜索新闻
            - "all": 搜索所有

    Returns:
        搜索结果
    """
    # 模拟数据库
    products = {
        "手机": "iPhone 15 (¥5999), 小米14 (¥3999), 华为Mate60 (¥6999)",
        "笔记本": "MacBook Pro (¥12999), ThinkPad X1 (¥9999), 华为MateBook (¥7999)",
        "耳机": "AirPods Pro (¥1999), Sony WH-1000XM5 (¥2499)"
    }

    news = {
        "AI": "1. GPT-5 即将发布  2. AI 芯片市场增长 30%  3. 新AI法规出台",
        "科技": "1. 量子计算新突破  2. 6G 技术测试  3. 新能源汽车销量创新高"
    }

    results = []

    if category in ["product", "all"]:
        for key, value in products.items():
            if keyword in key:
                results.append(f"【产品】{key}：{value}")

    if category in ["news", "all"]:
        for key, value in news.items():
            if keyword in key or keyword in value:
                results.append(f"【新闻】{key} 相关：{value}")

    if results:
        return "\n".join(results)
    else:
        return f"未找到关于 '{keyword}' 的{category}信息"
