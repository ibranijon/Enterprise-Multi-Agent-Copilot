from graph.agents.AGENT_Planner import planner_agent

if __name__ == "__main__":
    user_prompt = (
        "Elaborate on hearfailure and causes of it and propose mitigation steps."
        
    )

    plan = planner_agent.invoke({"question": user_prompt})

    print("\n=== PLANNER AGENT OUTPUT ===\n")
    print(plan)