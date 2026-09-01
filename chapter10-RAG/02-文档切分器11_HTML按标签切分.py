from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import MiniMaxEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# 加载文本
with open("../asset/load/09-ai1.txt", encoding="utf-8") as f:
    state_of_the_union = f.read()  #返回字符串

# 获取嵌入模型 —— minimax 嵌入模型: embo-01, 维度 1536.
#    api_key / group_id 都不显式传, MiniMaxEmbeddings 内部会自动从 env 读:
#        MINIMAX_API_KEY       —— minimax 控制台 → 订阅 Key
#        MINIMAX_GROUP_ID      —— minimax 控制台 → 基本信息 → 团队 ID
#    注: langchain 的 validate_environment 用 `if value := data.get(k)` 做空值过滤,
#    直接传空字符串会被跳过并报错;传非空字符串又会被强校验 group 与 key 是否匹配.
#    让它从 env 读是唯一稳妥路径 —— 个人开发者只要在 .env 配 MINIMAX_GROUP_ID 即可.
embedding_model = MiniMaxEmbeddings(
    model="embo-01",
)

# embedding_model = OpenAIEmbeddings(
#     model="BAAI/bge-m3", # 付费模型 ID： Pro/BAAI/bge-m3
#     base_url=os.getenv("SILICONFLOW_BASE_URL"),
#     api_key=os.getenv("SILICONFLOW_API_KEY"),
#     dimensions=1024
# )


# 获取切割器
text_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile", # 断点阈值类型：字面值["百分位数", "标准差", "四分位距", "梯度"] 选其一
    breakpoint_threshold_amount=65.0, # 断点阈值数量 (极低阈值 → 高分割敏感度)
    sentence_split_regex=r"(?<=[。？！])\s+" # 句子切分正则:遇到中文的句号、感叹号、问号（。？！）且后面带有空格时，先将其切分为独立的“句子”。
)

# 切分文档
docs = text_splitter.create_documents(texts = [state_of_the_union])

print(len(docs))
for doc in docs:
    print(f"🔍 文档: {doc}")