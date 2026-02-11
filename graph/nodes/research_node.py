from typing import Any, Dict
from graph.state import GraphState
from graph.agents.AGENT_Planner import planner_agent

def planner_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    plan = planner_agent.invoke({"question": question})
    return {"question": question, "plan": plan}
