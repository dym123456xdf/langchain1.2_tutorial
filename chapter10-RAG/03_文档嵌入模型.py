from langchain_community.embeddings import MiniMaxEmbeddings
import os
from dotenv import load_dotenv


load_dotenv(override=True)


# 获取嵌入模型 —— minimax 嵌入模型: embo-01, 维度 1536.
#    api_key / group_id 都不显式传, MiniMaxEmbeddings 内部会自动从 env 读:
#        MINIMAX_API_KEY       —— minimax 控制台 → 订阅 Key
#        MINIMAX_GROUP_ID      —— minimax 控制台 → 基本信息 → 团队 ID
embedding_model = MiniMaxEmbeddings(
    model="embo-01",
)


text = "你好，很高兴认识你"


embed_docs = embedding_model.embed_query(text)

print(len(embed_docs))

print(embed_docs[:5])