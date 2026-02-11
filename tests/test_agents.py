from graph.agents.AGENT_Planner import planner_agent


def test_planner_agent_returns_bullets():
    question = "Explain risks of AI in healthcare and propose mitigation steps."

    output = planner_agent.invoke({"question": question})

    assert output is not None, "Planner returned None"
    assert isinstance(output, str), "Planner output must be a string"

    lines = [line.strip() for line in output.splitlines() if line.strip()]

    assert len(lines) > 0, "Planner returned empty output"
    assert len(lines) <= 5, f"Planner returned more than 5 bullets: {len(lines)}"

    for line in lines:
        assert line.startswith("- "), f"Line is not a bullet point: {line}"