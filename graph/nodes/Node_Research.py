from typing import Dict, Any
from graph.state import GraphState
from graph.agents.AGENT_Research import research_agent


def research_node(state: GraphState) -> Dict[str, Any]:
    queries = state.get("plan", [])  # planner outputs List[str]

    # research_agent already retrieves ~3 chunks/query via retriever configuration
    result = research_agent(queries)

    return {
        "question": state.get("question", ""),
        "plan": queries,
        "documents": result.get("documents", []),
        "research_trace": result.get("trace", {}),
        "document_relevancy": bool(result.get("documents", [])),
    }
