from graph.agents.AGENT_Planner import planner_agent
from graph.agents.AGENT_Research import research_agent

if __name__ == "__main__":
    user_question = (
        "Talk about heart failure and the process of hospital readmission "
        "and discuss ways to reduce readmissions."
    )

    print("\n=== USER QUESTION ===\n")
    print(user_question)

    # 1) Planner Agent
    plan = planner_agent.invoke({"question": user_question})

    print("\n=== PLANNER OUTPUT (queries) ===\n")
    for i, q in enumerate(plan, 1):
        print(f"{i}. {q}")

    # 2) Research Agent
    result = research_agent(plan)

    docs = result["documents"]
    trace = result["trace"]

    print("\n=== RESEARCH SUMMARY ===")
    print(f"Queries used: {len(trace['queries'])}")
    print(f"Chunks kept: {trace['kept']}")
    print(f"Chunks dropped: {trace['dropped']}")

    print("\n=== TRACE (per chunk) ===")
    for row in trace["rows"]:
        print(
            f"[{row['grade'].upper()}] "
            f"query='{row['query']}' "
            f"source='{row['source']}'"
        )

    print("\n=== KEPT CHUNKS (preview) ===")
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "unknown")
        matched = (d.metadata or {}).get("matched_query", "")
        preview = (d.page_content or "").replace("\n", " ").strip()
        preview = preview[:250] + ("..." if len(preview) > 250 else "")

        print(f"\n#{i}")
        print(f"Source: {src}")
        print(f"Matched query: {matched}")
        print(preview)
