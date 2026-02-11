from typing import Any, Dict, List
from graph.state import GraphState
from graph.agents.AGENT_Verifier import verify_draft
from langchain_core.documents import Document


def _sources_block(documents: List[Document], cited_nums: List[int]) -> str:
    if not cited_nums:
        return "Not found in sources."
    seen = set()
    lines: List[str] = ["Sources"]
    for n in cited_nums:
        if n in seen:
            continue
        seen.add(n)
        idx = n - 1
        if idx < 0 or idx >= len(documents):
            continue
        md = documents[idx].metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page_start", md.get("page"))
        chunk = md.get("chunk_id", "?")
        lines.append(f"- {src} (page {page}, chunk {chunk})")
    return "\n".join(lines) if len(lines) > 1 else "Not found in sources."


def verifier_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    draft = state.get("writer_draft", {})

    verified = verify_draft(question=question, documents=documents, draft=draft)

    if verified.get("invalid"):
        return {
            **state,
            "final_output": verified["invalid"],
            "sources": "Not found in sources.",
            "citations_used": [],
            "verifier_issues": verified.get("issues", []),
        }

    sources = _sources_block(documents, verified.get("citations_used", []))

    email_display = (
        f"To: {verified.get('email_to','[EMAIL]')}\n"
        f"Subject: {verified.get('email_subject','')}\n\n"
        f"{verified.get('email_body','')}"
    )

    return {
        **state,
        "executive_summary": verified.get("executive_summary", "").strip(),
        "email_to": verified.get("email_to", "[EMAIL]"),
        "email_subject": verified.get("email_subject", "").strip(),
        "email_body": verified.get("email_body", "").strip(),
        "email": email_display,
        "actions": verified.get("actions", []),
        "citations_used": verified.get("citations_used", []),
        "sources": sources,
        "verifier_issues": verified.get("issues", []),
    }
