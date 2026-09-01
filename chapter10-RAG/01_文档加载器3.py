from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="../asset/load/02-load.csv",
)

docs = loader.load()
print(docs)