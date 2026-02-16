import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_API_KEY")
)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


PERSIST_DIR = "./retrival/chroma"
COLLECTION = "rag-chroma"


llm = OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPENAI_API_KEY)


retriever = Chroma(
    collection_name=COLLECTION,
    embedding_function=llm,
    persist_directory=PERSIST_DIR,
).as_retriever(search_kwargs={"k": 2})   # 2 Chunks per query