from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional
import os
import json
import uuid


def new_run_id() -> str:
    return uuid.uuid4().hex


def trace_event(
    state: Dict[str, Any],
    node: str,
    event: str,
    data: Optional[Dict[str, Any]] = None,
    write_jsonl: bool = True,
    jsonl_path: str = "logs/run.jsonl",
) -> None:
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "run_id": state.get("run_id"),
        "node": node,
        "event": event,  # "start" | "end" | "error"
        "data": data or {},
    }

    trace = state.get("trace") or []
    trace.append(record)
    state["trace"] = trace

    if write_jsonl:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
