from typing import Dict, Any
from graph.state import GraphState
from graph.agents.AGENT_Research import research_agent


def research_node(state: GraphState) -> Dict[str, Any]:
    queries = state.get("plan", [])  # planner outputs List[str]
    result = research_agent(queries=queries, k_per_query=3)

    return {
        "question": state.get("question", ""),
        "plan": queries,
        "documents": result["documents"],
        "research_trace": result["trace"],
        "document_relevancy": bool(result["documents"]),
    }
