import os
import streamlit as st
from dotenv import load_dotenv

def get_openai_key():
    load_dotenv()
    api_key = (
        os.getenv("OPENAI_API_KEY") or 
        st.secrets.get("OPENAI_API_KEY") or 
        os.environ.get("OPENAI_API_KEY")
    )
    
    return api_key

api_key = get_openai_key()


from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


PERSIST_DIR = "./retrival/chroma"
COLLECTION = "rag-chroma"


llm = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = Chroma(
    collection_name=COLLECTION,
    embedding_function=llm,
    persist_directory=PERSIST_DIR,
).as_retriever(search_kwargs={"k": 2})   # 2 Chunks per query