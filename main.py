from graph.graph_flow import app
from graph.utils.tracing import new_run_id


def format_final_output(state: dict) -> str:
    if "final_output" in state and state["final_output"]:
        return state["final_output"]

    executive_summary = state.get("executive_summary", "").strip()
    email = state.get("email", "").strip()
    actions = state.get("actions", []) or []
    sources = state.get("sources", "").strip()

    out = []

    out.append("EXECUTIVE SUMMARY (max 150 words)")
    out.append(executive_summary if executive_summary else "Not found in sources.")
    out.append("")

    out.append("CLIENT-READY EMAIL")
    out.append(email if email else "Not found in sources.")
    out.append("")

    out.append("ACTION LIST (owner, due date, confidence)")
    if not actions:
        out.append("Not found in sources.")
    else:
        for i, a in enumerate(actions, 1):
            task = a.get("task", "")
            owner = a.get("owner", "")
            due = a.get("due_date", "")
            conf = a.get("confidence", "")
            out.append(f"{i}. {task} (Owner: {owner}, Due: {due}, Confidence: {conf})")
    out.append("")

    out.append("SOURCES AND CITATIONS")
    out.append(sources if sources else "Not found in sources.")

    return "\n".join(out)


def main():
    user_input = "Based on published evidence, what post-discharge interventions are most effective at reducing heart failure readmissions and what concrete next steps should a hospital take? Send this to Recipient: Director of Care Transitions transition@org.com"

    final_state = app.invoke(
        {
            "question": user_input,
            "run_id": new_run_id(),
            "trace": [],
        }
    )

    print(format_final_output(final_state))

    # Optional: quick trace summary in console
    # (full trace is also written to logs/run.jsonl if you used the tracing helper I gave you)
    if final_state.get("trace"):
        print("\n--- TRACE SUMMARY ---")
        for e in final_state["trace"]:
            print(f"{e['ts']} | {e['node']} | {e['event']} | {e.get('data', {})}")


if __name__ == "__main__":
    main()
