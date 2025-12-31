from langchain_community.document_loaders import TextLoader

loader = TextLoader("./document-loader/doc.txt")  
content = loader.load()
print(content)