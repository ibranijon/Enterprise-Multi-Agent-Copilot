from typing import Any, Dict
from graph.state import GraphState
from graph.agents.AGENT_Writer import write_draft
from graph.utils.tracing import trace_event

NODE = "writer"


def writer_node(state: GraphState) -> Dict[str, Any]:
    docs = state.get("documents", []) or []
    trace_event(state, NODE, "start", {"documents": len(docs)})

    try:
        question = state["question"]
        documents = state.get("documents", [])

        draft = write_draft(question=question, documents=documents)

        trace_event(
            state,
            NODE,
            "end",
            {
                "invalid": bool(draft.get("invalid")) if isinstance(draft, dict) else None,
                "citations_used": draft.get("citations_used") if isinstance(draft, dict) else None,
                "actions": len(draft.get("actions", [])) if isinstance(draft, dict) else None,
            },
        )

        return {
            **state,
            "writer_draft": draft,
        }

    except Exception as e:
        trace_event(state, NODE, "error", {"error": repr(e)})
        raise
