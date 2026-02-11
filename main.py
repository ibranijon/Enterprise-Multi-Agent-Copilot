from graph.agents.AGENT_Planner import planner_agent
from graph.agents.AGENT_Research import research_agent
from graph.agents.AGENT_Writer import write_draft
from graph.agents.AGENT_Verifier import verify_draft


def _sources_block(documents, cited_nums):
    if not cited_nums:
        return "Not found in sources."

    seen = set()
    lines = ["Sources"]
    for n in cited_nums:
        if n in seen:
            continue
        seen.add(n)
        idx = n - 1
        if idx < 0 or idx >= len(documents):
            continue
        md = documents[idx].metadata or {}
        src = md.get("source", "unknown")
        page = md.get("page_start", md.get("page"))
        chunk = md.get("chunk_id", "?")
        lines.append(f"- {src} (page {page}, chunk {chunk})")

    return "\n".join(lines) if len(lines) > 1 else "Not found in sources."


def main():
    question = (
        "Based on published evidence, what post-discharge interventions are most effective at reducing heart failure readmissions, "
        "and what concrete next steps should a hospital take? Send this to Recipient: Director of Care Transitions transition@org.com"
    )

    state = {"question": question}

    # 1) Planner
    plan = planner_agent.invoke({"question": state["question"]})
    state["plan"] = plan

    print("\n=== PLANNER OUTPUT ===\n")
    for i, q in enumerate(plan, 1):
        print(f"{i}. {q}")

    # 2) Research
    research_out = research_agent(state["plan"])
    state["documents"] = research_out.get("documents", [])
    state["research_trace"] = research_out.get("trace", {})
    state["document_relevancy"] = bool(state["documents"])

    print("\n=== RESEARCH SUMMARY ===")
    queries = state["research_trace"].get("queries", [])
    print(f"Queries used: {len(queries)}")
    print(f"Chunks kept: {state['research_trace'].get('kept')}")
    print(f"Chunks dropped: {state['research_trace'].get('dropped')}")

    # NOTE: do NOT stop early anymore — Writer/Verifier handle invalid output policy.
    # 3) Draft (Writer)
    draft_out = write_draft(
        question=state["question"],
        documents=state["documents"],
    )
    state["writer_draft"] = draft_out

    # 4) Verify (Verifier)
    verified_out = verify_draft(
        question=state["question"],
        documents=state["documents"],
        draft=state["writer_draft"],
    )

    # If verifier (or writer) says invalid → print ONLY one sentence
    if verified_out.get("invalid"):
        state["generation"] = verified_out["invalid"]
        print("\n=== FINAL OUTPUT (generation) ===\n")
        print(state["generation"])
        return

    # Build sources from citations_used (writer-chosen subset, verifier cannot add new ones)
    citations_used = verified_out.get("citations_used", []) or []
    sources_block = _sources_block(state["documents"], citations_used)

    email_display = (
        f"To: {verified_out.get('email_to', '[EMAIL]')}\n"
        f"Subject: {verified_out.get('email_subject', '')}\n\n"
        f"{verified_out.get('email_body', '')}"
    )

    # Assemble final output into "generation"
    generation = (
        f"EXECUTIVE SUMMARY\n{verified_out.get('executive_summary','')}\n\n"
        f"{sources_block}\n\n"
        f"CLIENT-READY EMAIL\n{email_display}\n\n"
        f"ACTION LIST\n"
    )

    actions = verified_out.get("actions", []) or []
    for i, a in enumerate(actions, 1):
        generation += (
            f"{i}. {a.get('task','')}"
            f" (Owner: {a.get('owner','')}, Due: {a.get('due_date','')}, "
            f"Confidence: {a.get('confidence','')})\n"
        )

    # Optional: print verifier issues if any (debug)
    issues = verified_out.get("issues", [])
    if issues:
        generation += "\nVERIFIER NOTES\n"
        for j, issue in enumerate(issues, 1):
            generation += f"- {issue}\n"

    state["generation"] = generation

    print("\n=== FINAL OUTPUT (generation) ===\n")
    print(state["generation"])


if __name__ == "__main__":
    main()
