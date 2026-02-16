from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()
from langchain_openai import OpenAIEmbeddings


PERSIST_DIR = "./retrival/chroma"
COLLECTION = "rag-chroma"


llm = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = Chroma(
    collection_name=COLLECTION,
    embedding_function=llm,
    persist_directory=PERSIST_DIR,
).as_retriever(search_kwargs={"k": 2})   # 2 Chunks per query