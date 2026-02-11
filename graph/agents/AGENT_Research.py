from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document

from retrival.doc_retriver import retriever
from graph.chains.chunk_grader import retrieval_grader


def _dedupe_docs(docs: List[Document]) -> List[Document]:
    seen: set[Tuple[str, str]] = set()
    out: List[Document] = []
    for d in docs:
        content = (d.page_content or "").strip()
        src = str((d.metadata or {}).get("source", ""))
        key = (src, content)
        if content and key not in seen:
            seen.add(key)
            out.append(d)
    return out


def research_agent(queries: List[str]) -> Dict[str, Any]:
    """
    Research Agent
    - Receives up to 5 queries from Planner
    - For each query, calls Chroma retriever (already configured to return 3 chunks/query)
    - Runs strict relevance grading on each chunk against the SAME query
    - Returns only chunks graded 'yes'
    """
    if not queries:
        return {"documents": [], "trace": {"queries": [], "kept": 0, "dropped": 0, "rows": []}}

    queries = [q.strip() for q in queries[:5] if q and q.strip()]

    kept_docs: List[Document] = []
    dropped = 0
    trace_rows: List[Dict[str, Any]] = []

    for q in queries:
        retrieved: List[Document] = retriever.invoke(q)  # expects ~3 docs now (k=3)

        for d in retrieved:
            text = (d.page_content or "").strip()
            if not text:
                dropped += 1
                continue

            grade = retrieval_grader.invoke({"question": q, "document": text}).binary_score
            is_yes = str(grade).strip().lower() == "yes"

            trace_rows.append(
                {
                    "query": q,
                    "grade": "yes" if is_yes else "no",
                    "source": (d.metadata or {}).get("source", "unknown"),
                }
            )

            if is_yes:
                md = dict(d.metadata or {})
                md["matched_query"] = q
                kept_docs.append(Document(page_content=text, metadata=md))
            else:
                dropped += 1

    kept_docs = _dedupe_docs(kept_docs)

    return {
        "documents": kept_docs,
        "trace": {
            "queries": queries,
            "kept": len(kept_docs),
            "dropped": dropped,
            "rows": trace_rows,
        },
    }
