from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "./retrival/chroma"
COLLECTION = "rag-chroma"

llm = OllamaEmbeddings(model="nomic-embed-text:latest")

retriever = Chroma(
    collection_name=COLLECTION,
    embedding_function=llm,
    persist_directory=PERSIST_DIR,
).as_retriever(search_kwargs={"k": 5})   # This returns 5 chunks