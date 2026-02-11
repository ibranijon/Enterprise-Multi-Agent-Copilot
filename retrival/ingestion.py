from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Configuration


@dataclass(frozen=True)
class IngestionConfig:


    dataset_dir: Path = Path("./data")
    persist_directory: Path = Path("./retrival/chroma")
    collection_name: str = "rag-chroma"
    embedding_model: str = "nomic-embed-text:latest"

    # Chunking (approximate tokens; enforced by a hard char cap as well)
    chunk_size_tokens: int = 1800
    chunk_overlap_tokens: int = 100
    max_chunk_chars: int = 2000

    # Header/footer removal (repeated lines across pages)
    header_footer_window_lines: int = 3
    repeat_line_min_len: int = 8
    repeat_line_ratio: float = 0.6

    manifest_path: Path = Path("./ingestion_manifest.json")



# Utilities


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_text(text: str) -> str:
    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # De-hyphenate common PDF line breaks: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Clean whitespace/newlines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()



# PDF Loading (PyMuPDF first, pdfplumber fallback)


def _load_pdf_pages_pdfplumber(pdf_path: Path) -> List[Document]:
    import pdfplumber

    docs: List[Document] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"page": i}))
    return docs


def _load_pdf_pages(pdf_path: Path) -> List[Document]:
   
    try:
        docs = PyMuPDFLoader(str(pdf_path)).load()
    except Exception:
        docs = []

    # If extraction is weak/empty, fallback to pdfplumber
    total_chars = sum(len((d.page_content or "").strip()) for d in docs) if docs else 0
    if not docs or total_chars < 200:
        docs = _load_pdf_pages_pdfplumber(pdf_path)

    return docs


# Header/Footer Removal 


def _collect_repeated_lines(pages: List[str], cfg: IngestionConfig) -> set[str]:
    candidates: List[str] = []
    w = cfg.header_footer_window_lines

    for t in pages:
        lines = [ln.strip() for ln in t.splitlines()]
        head = [ln for ln in lines[:w] if len(ln) >= cfg.repeat_line_min_len]
        tail = [ln for ln in lines[-w:] if len(ln) >= cfg.repeat_line_min_len]
        candidates.extend(head)
        candidates.extend(tail)

    counts = Counter(candidates)
    threshold = max(2, int(len(pages) * cfg.repeat_line_ratio))
    return {ln for ln, c in counts.items() if c >= threshold}


def _strip_repeated_lines(page_text: str, repeated: set[str]) -> str:
    out: List[str] = []
    for ln in page_text.splitlines():
        s = ln.strip()
        if s and s in repeated:
            continue
        out.append(ln)
    return "\n".join(out)

# Cleaning + Metadata Enrichment


def _prepare_docs(pdf_path: Path, cfg: IngestionConfig) -> List[Document]:
    # Load per-page documents (keeps page metadata for citations)
    pages = _load_pdf_pages(pdf_path)
    raw_texts = [p.page_content or "" for p in pages]

    # Remove repeating header/footer lines when we have enough pages to detect patterns
    repeated = _collect_repeated_lines(raw_texts, cfg) if len(raw_texts) >= 3 else set()

    cleaned: List[Document] = []
    for p in pages:
        text = p.page_content or ""
        if repeated:
            text = _strip_repeated_lines(text, repeated)
        text = _normalize_text(text)

        meta = dict(p.metadata or {})
        meta["source"] = pdf_path.name
        meta["source_path"] = str(pdf_path.as_posix())

        # Ensure page is int when present
        if meta.get("page") is not None:
            try:
                meta["page"] = int(meta["page"])
            except Exception:
                meta["page"] = None

        cleaned.append(Document(page_content=text, metadata=meta))

    return cleaned



# Chunking + Stable IDs for Citations


def _enforce_max_chars(docs: List[Document], max_chars: int) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if not txt:
            continue

        if len(txt) <= max_chars:
            out.append(d)
            continue

        start = 0
        while start < len(txt):
            end = min(len(txt), start + max_chars)
            chunk_txt = txt[start:end].strip()
            if chunk_txt:
                out.append(Document(page_content=chunk_txt, metadata=dict(d.metadata)))
            start = end

    return out


def _split_and_tag(docs: List[Document], cfg: IngestionConfig) -> Tuple[List[Document], List[str]]:
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=cfg.chunk_size_tokens,
        chunk_overlap=cfg.chunk_overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    splits = splitter.split_documents(docs)
    splits = _enforce_max_chars(splits, max_chars=cfg.max_chunk_chars)

    
    per_page_counter: Dict[Tuple[str, int], int] = defaultdict(int)
    ids: List[str] = []

    for d in splits:
        src = str(d.metadata.get("source", "unknown"))

        page = d.metadata.get("page", None)
        if page is not None:
            try:
                page = int(page)
            except Exception:
                page = None

        page_for_key = page if page is not None else -1

        key = (src, page_for_key)
        chunk_id = per_page_counter[key]
        per_page_counter[key] += 1

        source_path = str(d.metadata.get("source_path", src))
        doc_id = _sha1(source_path)
        content_hash = _sha1(d.page_content)

       
        d.metadata["doc_id"] = doc_id
        d.metadata["chunk_id"] = chunk_id
        d.metadata["content_hash"] = content_hash
        d.metadata["page_start"] = page
        d.metadata["page_end"] = page

       
        stable_id = f"{doc_id}::p{page_for_key}::c{chunk_id}"
        ids.append(stable_id)

    return splits, ids



# Main Ingestion Pipeline


def ingest(cfg: IngestionConfig = IngestionConfig()) -> dict:
    if not cfg.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {cfg.dataset_dir.resolve()}")

    pdf_files = sorted(p for p in cfg.dataset_dir.rglob("*.pdf") if p.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under: {cfg.dataset_dir.resolve()}")

    all_docs: List[Document] = []
    failures: List[dict] = []

   
    for pdf in pdf_files:
        try:
            docs = _prepare_docs(pdf, cfg)
            docs = [d for d in docs if (d.page_content or "").strip()]
            all_docs.extend(docs)
        except Exception as e:
            failures.append({"file": str(pdf), "error": repr(e)})

    
    splits, ids = _split_and_tag(all_docs, cfg)

    if not splits:
        manifest = {
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "dataset_dir": str(cfg.dataset_dir.resolve()),
            "collection_name": cfg.collection_name,
            "persist_directory": str(cfg.persist_directory.resolve()),
            "embedding_model": cfg.embedding_model,
            "chunk_size_tokens": cfg.chunk_size_tokens,
            "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
            "max_chunk_chars": cfg.max_chunk_chars,
            "pdf_count": len(pdf_files),
            "page_docs_count": len(all_docs),
            "chunk_count": 0,
            "failures": failures,
            "error": "No chunks created (PDF extraction returned empty text).",
        }
        cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # 3) Embed + store in Chroma
    embeddings = OllamaEmbeddings(model=cfg.embedding_model)
    vectorstore = Chroma.from_documents(
        documents=splits,
        ids=ids,
        collection_name=cfg.collection_name,
        embedding=embeddings,
        persist_directory=str(cfg.persist_directory),
    )

    # Persist (some versions auto-persist; this keeps it explicit)
    try:
        vectorstore.persist()
    except Exception:
        pass

    manifest = {
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "dataset_dir": str(cfg.dataset_dir.resolve()),
        "collection_name": cfg.collection_name,
        "persist_directory": str(cfg.persist_directory.resolve()),
        "embedding_model": cfg.embedding_model,
        "chunk_size_tokens": cfg.chunk_size_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "max_chunk_chars": cfg.max_chunk_chars,
        "pdf_count": len(pdf_files),
        "page_docs_count": len(all_docs),
        "chunk_count": len(splits),
        "failures": failures,
    }

    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = ingest()
    print(json.dumps(result, indent=2))