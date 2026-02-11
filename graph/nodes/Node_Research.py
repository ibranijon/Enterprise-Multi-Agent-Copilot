from typing import Dict, Any
from graph.state import GraphState
from graph.agents.AGENT_Research import research_agent
from graph.utils.tracing import trace_event

NODE = "research"


def research_node(state: GraphState) -> Dict[str, Any]:
    queries = state.get("plan", []) or []
    trace_event(state, NODE, "start", {"queries": len(queries)})

    try:
        result = research_agent(queries)

        documents = result.get("documents", []) or []
        trace = result.get("trace", {}) or {}

        trace_event(
            state,
            NODE,
            "end",
            {
                "documents": len(documents),
                "kept": trace.get("kept"),
                "dropped": trace.get("dropped"),
                "sources": list({(d.metadata or {}).get("source") for d in documents})[:5],
            },
        )

        return {
            **state,
            "plan": queries,
            "documents": documents,
            "research_trace": trace,
            "document_relevancy": bool(documents),
        }

    except Exception as e:
        trace_event(state, NODE, "error", {"error": repr(e)})
        raise
