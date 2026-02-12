import sys
from pathlib import Path
import uuid
import traceback
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assets.UI import apply_page_config, apply_css
from assets.Components import (
    render_header,
    render_structured_output,
    render_invalid,
    render_runtime_error,
)

try:
    from graph.graph_flow import app as langgraph_app
except Exception as e:
    langgraph_app = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


def _build_input_state(question: str) -> dict:
    return {
        "question": question,
        "trace": [],
        "run_id": str(uuid.uuid4()),
    }


def _is_invalid(state: dict) -> bool:
    return bool(state.get("final_output"))


def main() -> None:
    apply_page_config()
    apply_css()

    render_header()

    if IMPORT_ERROR is not None:
        render_runtime_error(
            "Could not import LangGraph app from graph/graph_flow.py",
            details=str(IMPORT_ERROR),
        )
        st.stop()

    if langgraph_app is None:
        render_runtime_error("LangGraph app is not available.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your request…")

    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.status("Running agents…", expanded=False):
                try:
                    out_state = langgraph_app.invoke(_build_input_state(user_prompt))
                except Exception as ex:
                    render_runtime_error(
                        "The run crashed (exception). This is different from an INVALID run (which is signaled via `final_output`).",
                        details="".join(
                            traceback.format_exception(type(ex), ex, ex.__traceback__)
                        ),
                    )
                    st.stop()

            if _is_invalid(out_state):
                assistant_md = render_invalid(out_state.get("final_output", "Invalid."))
            else:
                assistant_md = render_structured_output(out_state)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_md})
        st.rerun()


if __name__ == "__main__":
    main()
