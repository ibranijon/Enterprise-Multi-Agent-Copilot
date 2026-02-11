from graph.agents.AGENT_Planner import planner_agent
from graph.agents.AGENT_Research import research_agent
from graph.agents.AGENT_Writer import writer_agent


def main():
    question = (
        "Talk about heart failure and the process of hospital readmission and discuss ways to reduce readmissions. Send the email to operations@hospital.org."
    )

    state = {
        "question": question
    }

    # 1) Planner
    plan = planner_agent.invoke({"question": state["question"]})
    state["plan"] = plan

    print("\n=== PLANNER OUTPUT ===\n")
    for i, q in enumerate(plan, 1):
        print(f"{i}. {q}")

    # 2) Research
    research_out = research_agent(state["plan"])
    state["documents"] = research_out["documents"]
    state["research_trace"] = research_out["trace"]
    state["document_relevancy"] = bool(state["documents"])

    print("\n=== RESEARCH SUMMARY ===")
    print(f"Queries used: {len(state['research_trace']['queries'])}")
    print(f"Chunks kept: {state['research_trace']['kept']}")
    print(f"Chunks dropped: {state['research_trace']['dropped']}")

    # If no evidence, stop early (matches your Writer failure requirement)
    if not state["document_relevancy"]:
        print("\nNo relevant evidence found. Stopping before Draft.")
        return

    # 3) Draft (Writer)
    draft_out = writer_agent(
        question=state["question"],
        documents=state["documents"],
        model_name="gpt-4o-mini",
    )

    # Assemble final output into your GraphState "generation"
    generation = (
        f"EXECUTIVE SUMMARY\n{draft_out['executive_summary']}\n\n"
        f"{draft_out['sources']}\n\n"
        f"CLIENT-READY EMAIL\n{draft_out['email']}\n\n"
        f"ACTION LIST\n"
    )

    for i, a in enumerate(draft_out["actions"], 1):
        generation += f"{i}. {a['task']} (Owner: {a['owner']}, Due: {a['due_date']}, Confidence: {a['confidence']})\n"

    state["generation"] = generation

    print("\n=== FINAL OUTPUT (generation) ===\n")
    print(state["generation"])


if __name__ == "__main__":
    main()
