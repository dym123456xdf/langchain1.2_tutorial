from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    # 文件路径，支持本地文件和在线文件链接
    # file_path="../asset/load/04-sample.pdf",
    file_path="https://arxiv.org/pdf/alg-geom/9202012",
    # 提取模式:控制如何从 PDF 文件中解析和提取文本结构。
    #   plain 提取文本，默认值
    #   layout 布局感知提取模式，通常会通过插入大量的空格、换行符，来模拟原文档中的多栏、缩进和间距（适用场景：学术论文（如 arXiv 论文）、多栏报刊杂志、带有左右分栏的合同）
    extraction_mode="plain",
)

docs = loader.load()

print(docs)
print(len(docs))