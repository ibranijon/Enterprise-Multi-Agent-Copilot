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


def ensure_vectorstore_ready() -> None:
    vs = _vectorstore()
    if _count(vs) > 0:
        return

    st.warning("Vector store not found on this deployment. Build it now by uploading PDFs or using ./data.")

    uploaded = st.file_uploader("Upload PDFs to build the vector store", type=["pdf"], accept_multiple_files=True)

    pdf_paths = []
    if uploaded:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            out = DATASET_DIR / f.name
            out.write_bytes(f.getvalue())
            pdf_paths.append(out)

    has_local_data = DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.pdf"))

    if not uploaded and not has_local_data:
        st.info("No PDFs available. Upload PDFs above, or add PDFs under ./data in the repo.")
        st.stop()

    with st.spinner("Building Chroma vector store..."):
        cfg = IngestionConfig(
            dataset_dir=DATASET_DIR,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION,
            embedding_model="text-embedding-3-small",
        )
        ingest(cfg)

    st.success("Vector store built successfully. Reloading...")
    st.rerun()


# Make sure DB exists before exposing retriever
ensure_vectorstore_ready()

retriever = _vectorstore().as_retriever(search_kwargs={"k": 2})
