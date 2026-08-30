"""
@Author:dym
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, cast
import os

# ============================================================
# 1、读取 .env 配置 (.env 里的变量优先级最高, override=True 会覆盖 shell 变量)
# ============================================================
load_dotenv(override=True)

# ============================================================
# 2、参数校验: 防止 .env 缺字段时给一个晦涩的 ValidationError
# ============================================================
AGNES_API_KEY  = os.getenv("AGNES_API_KEY")
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL")

if not AGNES_API_KEY:
    raise RuntimeError(
        "未找到环境变量 AGNES_API_KEY。请在项目根目录 .env 中配置:\n"
        "    AGNES_API_KEY=你的密钥"
    )
if not AGNES_BASE_URL:
    raise RuntimeError(
        "未找到环境变量 AGNES_BASE_URL。请在 .env 中配置, 例如:\n"
        "    AGNES_BASE_URL=https://apihub.agnes-ai.com/v1"
    )

# ============================================================
# 3、初始化模型 —— 用 init_chat_model(model_provider="openai") 走 OpenAI 协议,
#    配合 base_url + api_key 指向 agnes-ai, 是 agnes-ai 文档明说的接入方式.
#    注意: init_chat_model 默认走 ChatOpenAI, 兼容 OpenAI 协议的服务
#    (含 agnes-ai) 都能直接复用.
# ============================================================
model = init_chat_model(
    model="agnes-2.5-flash",                # agnes-ai 控制台上可见的 chat 模型名
    model_provider="openai",          # 走 OpenAI 协议 (agnes 兼容)
    api_key=AGNES_API_KEY,
    base_url=AGNES_BASE_URL,
)

# class Person(BaseModel):
#     """人物信息"""
#     name: str = Field(description="姓名")
#     age : int = Field(description="年龄")
#     occupation: str = Field(description="职业")
#
# # 创建结构化输出的大语言模型
# structured_model = model.with_structured_output(Person)
#
# result = structured_model.invoke("张三是一名30岁的软件工程师")
#
# print(result)
# print(type(result))
# print(f"姓名：{result.name}")
# print(f"年龄：{result.age}")
# print(f"职业：{result.occupation}")


# class MovieModel(BaseModel):
#     """电影的详细信息"""
#     title : str = Field(description="电影标题")
#     year : int = Field(description="发行年份")
#     director : str = Field(description="导演")
#     rating : float = Field(description="电影评分，满分十分")
#
#
# structured_model = model.with_structured_output(MovieModel)
#
# result = structured_model.invoke("给出电影盗梦空间的信息")
# print(result)

from pydantic import BaseModel, Field
# 定义输出结构
class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    sentiment: str = Field(description="情感倾向：positive/negative/neutral")
    confidence: float = Field(description="置信度，0-1之间")
    keywords: list[str] = Field(description="关键词列表")


# ✅ v1.x：使用 with_structured_output
structured_model = model.with_structured_output(SentimentAnalysis)

# 调用
text = "这个课程内容很实用，学到了很多知识，强烈推荐！"
result = structured_model.invoke(
    f"分析以下文本的情感：\n{text}"
)

print(f"类型: {type(result)}")  # <class 'SentimentAnalysis'>
print(f"情感: {result.sentiment}")
print(f"置信度: {result.confidence}")
print(f"关键词: {result.keywords}")