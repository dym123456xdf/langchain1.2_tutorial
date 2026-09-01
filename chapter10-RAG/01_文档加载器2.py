from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path="../asset/load/01-langchain-utf-8.txt",
    encoding="utf-8",
)

docs = loader.load()

print(docs)