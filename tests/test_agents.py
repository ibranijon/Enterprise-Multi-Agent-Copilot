# tests/test_agents.py
from graph.agents.AGENT_Planner import planner_agent


def test_planner_agent_returns_list_of_tasks():
    question = "Explain heartfailures and how to mitigate them."

    plan = planner_agent.invoke({"question": question})

    assert plan is not None, "Planner returned None"
    assert isinstance(plan, list), f"Planner must return a list, got {type(plan)}"
    assert 1 <= len(plan) <= 5, f"Planner must return 1–5 tasks, got {len(plan)}"

    for item in plan:
        assert isinstance(item, str), f"Each task must be a string, got {type(item)}"
        assert item.strip(), "Task string cannot be empty"
        assert not item.strip().startswith("- "), f"Tasks should not be bullet-formatted: {item}"
