from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="../asset/load/01-langchain-gbk.txt",
    encoding="gbk",
)

docs = loader.load()

print(docs)