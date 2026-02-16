import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it to Streamlit Secrets or local .env.")
    st.stop()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from retrival.ingestion import ingest, IngestionConfig

PERSIST_DIR = Path("./retrival/chroma")
COLLECTION = "rag-chroma"
DATASET_DIR = Path("./data")


@st.cache_resource
def _embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def _vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )


def _count(vs) -> int:
    try:
        return int(vs._collection.count())
    except Exception:
        return 0


def _has_pdfs(dataset_dir: Path) -> bool:
    return dataset_dir.exists() and any(dataset_dir.rglob("*.pdf"))


def ensure_vectorstore_ready() -> None:
    vs = _vectorstore()
    if _count(vs) > 0:
        return

    if not _has_pdfs(DATASET_DIR):
        st.error(
            "Vector store not found, and no PDFs were found in ./data.\n\n"
            "To run this app on Streamlit Cloud, include PDFs under ./data "
            "or build the vector store during deployment."
        )
        st.stop()

    with st.spinner("Building Chroma vector store from ./data ..."):
        cfg = IngestionConfig(
            dataset_dir=DATASET_DIR,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION,
            embedding_model="text-embedding-3-small",
        )
        ingest(cfg)

    st.success("Vector store built. Reloading...")
    st.rerun()


# Ensure DB exists before exposing retriever
ensure_vectorstore_ready()

retriever = _vectorstore().as_retriever(search_kwargs={"k": 2})
