import json
from pathlib import Path
from datetime import date
from graph.graph_flow import app
from graph.utils.tracing import new_run_id


PROMPTS_PATH = Path("eval/test_prompts/test_prompts.jsonl")


def _parse_date(s: str):
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def validate_output(result: dict, expect_invalid: bool) -> list[str]:
    errors: list[str] = []

    if expect_invalid:
        msg = result.get("final_output", "")
        if not msg or not isinstance(msg, str) or "Invalid" not in msg:
            errors.append("Expected invalid single-sentence output (final_output) but did not get it.")
        return errors

    if result.get("final_output"):
        errors.append("Expected valid output, but got final_output invalid message.")
        return errors

    exec_summary = (result.get("executive_summary") or "").strip()
    if not exec_summary:
        errors.append("Missing executive_summary.")
    else:
        if len(exec_summary.split()) > 170:
            errors.append("Executive summary seems too long (expected ~<=150 words).")

    email = (result.get("email") or "").strip()
    if not email or "To:" not in email or "Subject:" not in email:
        errors.append("Email missing or not formatted with To:/Subject:.")

    actions = result.get("actions") or []
    if not (2 <= len(actions) <= 4):
        errors.append(f"Expected 2-4 actions, got {len(actions)}.")

    for i, a in enumerate(actions, 1):
        for k in ("task", "owner", "due_date", "confidence"):
            if k not in a or not a.get(k):
                errors.append(f"Action {i} missing field: {k}")

        conf = a.get("confidence")
        if conf not in ("high", "medium", "low"):
            errors.append(f"Action {i} confidence invalid: {conf}")

        dd = _parse_date(a.get("due_date", ""))
        if dd is None:
            errors.append(f"Action {i} due_date not ISO YYYY-MM-DD: {a.get('due_date')}")
        else:
            if dd < date.today():
                errors.append(f"Action {i} due_date is in the past: {a.get('due_date')}")

    sources = (result.get("sources") or "").strip()
    if not sources or "Sources" not in sources:
        errors.append("Missing sources block.")
    elif "-" not in sources:
        errors.append("Sources block does not list any items.")

    citations_used = result.get("citations_used") or []
    if not citations_used:
        errors.append("Missing citations_used (expected at least one citation).")

    trace = result.get("trace") or []
    if not trace:
        errors.append("Missing trace events (state.trace empty).")
    else:
        nodes_seen = {e.get("node") for e in trace if isinstance(e, dict)}
        for needed in ("planner", "research", "writer", "verifier"):
            if needed not in nodes_seen:
                errors.append(f"Trace missing node events for: {needed}")

    return errors


def main():
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_PATH}")

    rows = []
    total = 0
    passed = 0

    for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        item = json.loads(line)
        qid = item["id"]
        question = item["question"]
        expect_invalid = bool(item.get("expect_invalid", False))

        try:
            result = app.invoke({"question": question, "run_id": new_run_id(), "trace": []})
            errors = validate_output(result, expect_invalid)
            ok = len(errors) == 0
        except Exception as e:
            ok = False
            errors = [f"Runtime exception: {repr(e)}"]

        rows.append((qid, ok, errors))
        if ok:
            passed += 1

    print(f"\nEVAL RESULTS: {passed}/{total} passed\n")
    for qid, ok, errors in rows:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {qid}")
        if errors:
            for err in errors[:10]:
                print(f"  - {err}")
        print("")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
