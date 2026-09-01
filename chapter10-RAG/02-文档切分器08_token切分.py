# 1.导入相关依赖
from langchain_text_splitters import TokenTextSplitter

# 2.初始化 TokenTextSplitter
text_splitter = TokenTextSplitter(
    chunk_size=33,  # 最大 token 数为 33
    chunk_overlap=0, # 重叠 token 数为 0
    # model_name="gpt-4", # 选择 GPT-4 模型的编码器
    encoding_name="cl100k_base",  # 使用 OpenAI 的编码器,将文本转换为 token 序列
)
# 3.定义文本
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。人工智能是指通过计算机程序模拟人类智能的一门科学。自20世纪50年代诞生以来，人工智能经历了多次起伏。"

# 4.开始切割
texts = text_splitter.split_text(text)

# 打印分割结果
print(f"原始文本被分割成了 {len(texts)} 个块:")
for i, chunk in enumerate(texts):
    print(f"块 {i+1}: 长度：{len(chunk)} 内容：{chunk}")
    print("-" * 50)