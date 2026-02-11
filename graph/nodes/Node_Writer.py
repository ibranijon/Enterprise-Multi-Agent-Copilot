from typing import Any, Dict
from graph.state import GraphState
from graph.agents.AGENT_Writer import writer_agent


def writer_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])

    out = writer_agent(question=question, documents=documents, model_name="llama3.1:latest")

    return {
        "question": question,
        "documents": documents,
        "executive_summary": out["executive_summary"],
        "email": out["email"],
        "actions": out["actions"],
        "sources": out["sources"],
    }
