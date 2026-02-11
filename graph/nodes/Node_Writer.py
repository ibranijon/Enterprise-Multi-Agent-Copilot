from typing import Any, Dict
from graph.state import GraphState
from graph.agents.AGENT_Writer import write_draft


def writer_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])

    draft = write_draft(question=question, documents=documents)

    return {
        **state,
        "writer_draft": draft,
    }
