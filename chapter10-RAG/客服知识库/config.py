# ==========================================
# 全局配置常量（被所有模块共享）
# 单一职责：存放可调参数。无任何 import 业务模块的逻辑。
# ==========================================

# ---------- Milvus ----------
MILVUS_URI = "http://localhost:19530"   # Milvus 服务地址
DB_NAME = "rag_tutorial"                # 数据库名
COLLECTION_NAME = "docs"                # 集合名（≈ 关系库的表）

# ---------- 知识文件 ----------
KNOWLEDGE_FILE = "../knowledge.txt"     # 相对本目录的本地知识文件

# ---------- 嵌入模型 ----------
# minimax embo-01 在官方文档中为 1024 维；与原 SiliconFlow BGE-M3 维度一致，milvus 集合无需重建。
EMBED_MODEL_NAME = "embo-01"
EMBED_DIM = 1024

# ---------- 文档切分 ----------
CHUNK_SIZE = 200
CHUNK_OVERLAP = 80
SEPARATORS = [
    "\n==============================\n",
    "\n\n",
    "\n",
    "。",
    " ",
    "",
]

# ---------- 检索 ----------
RETRIEVE_LIMIT = 3          # 默认检索返回片段数