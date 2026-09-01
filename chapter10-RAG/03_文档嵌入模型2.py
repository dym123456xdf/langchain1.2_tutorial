from langchain_community.embeddings import MiniMaxEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

# 获取嵌入模型 —— minimax 嵌入模型: embo-01, 维度 1536.
#    api_key / group_id 都不显式传, MiniMaxEmbeddings 内部会自动从 env 读:
#        MINIMAX_API_KEY       —— minimax 控制台 → 订阅 Key
#        MINIMAX_GROUP_ID      —— minimax 控制台 → 基本信息 → 团队 ID
embedding_model = MiniMaxEmbeddings(
    model="embo-01",
)

# 待嵌入的文本列表
texts = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
]

# 生成嵌入向量
embeded_docs = embedding_model.embed_documents(texts)

for i in range(len(texts)):
    print(f"{texts[i]}:{embeded_docs[i][:3]}",end="\n\n")