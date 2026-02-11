from typing import Any, Dict, List
from graph.state import GraphState
from graph.agents.AGENT_Verifier import verify_draft
from graph.utils.tracing import trace_event
from langchain_core.documents import Document

NODE = "verifier"


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
    trace_event(
        state,
        NODE,
        "start",
        {
            "documents": len(state.get("documents", []) or []),
            "has_draft": bool(state.get("writer_draft")),
        },
    )

    try:
        question = state["question"]
        documents = state.get("documents", [])
        draft = state.get("writer_draft", {})

        verified = verify_draft(question=question, documents=documents, draft=draft)

        if verified.get("invalid"):
            trace_event(
                state,
                NODE,
                "end",
                {"invalid": True, "issues": len(verified.get("issues", []) or [])},
            )
            return {
                **state,
                "final_output": verified["invalid"],
                "sources": "Not found in sources.",
                "citations_used": [],
                "verifier_issues": verified.get("issues", []),
            }

        citations_used = verified.get("citations_used", []) or []
        sources = _sources_block(documents, citations_used)

        actions = verified.get("actions", []) or []
        conf_counts = {
            "high": sum(1 for a in actions if a.get("confidence") == "high"),
            "medium": sum(1 for a in actions if a.get("confidence") == "medium"),
            "low": sum(1 for a in actions if a.get("confidence") == "low"),
        }

        trace_event(
            state,
            NODE,
            "end",
            {
                "invalid": False,
                "citations_used": citations_used,
                "actions": len(actions),
                "confidence": conf_counts,
                "issues": len(verified.get("issues", []) or []),
            },
        )

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
            "actions": actions,
            "citations_used": citations_used,
            "sources": sources,
            "verifier_issues": verified.get("issues", []),
        }

    except Exception as e:
        trace_event(state, NODE, "error", {"error": repr(e)})
        raise
