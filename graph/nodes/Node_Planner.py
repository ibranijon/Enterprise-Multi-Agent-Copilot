from typing import Dict, Any
from graph.state import GraphState
from graph.agents.AGENT_Planner import planner_agent
from graph.utils.tracing import trace_event

NODE = "planner"


def planner_node(state: GraphState) -> Dict[str, Any]:
    trace_event(state, NODE, "start", {"question_len": len(state.get("question", ""))})

    try:
        question = state["question"]
        plan = planner_agent.invoke({"question": question})

        trace_event(state, NODE, "end", {"plan_len": len(plan) if isinstance(plan, list) else None})

        return {
            **state,
            "question": question,
            "plan": plan,
        }

    except Exception as e:
        trace_event(state, NODE, "error", {"error": repr(e)})
        raise
