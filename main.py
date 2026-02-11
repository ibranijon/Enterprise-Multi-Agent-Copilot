from graph.agents.AGENT_Planner import planner_agent

if __name__ == "__main__":
    user_prompt = (
        "Talk about heart failures and process of readmission"
        
    )

    plan = planner_agent.invoke({"question": user_prompt})

    print("\n=== PLANNER AGENT OUTPUT ===\n")
    print(plan)