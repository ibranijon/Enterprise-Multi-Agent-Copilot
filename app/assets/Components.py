import streamlit as st
from typing import Any, Dict, List


def render_header() -> None:
    st.markdown("## Enterprise Multi-Agent Copilot")
    st.caption("LangGraph pipeline: Planner → Research → Writer → Verifier")


def _md_block(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"### {title}\n\n{body}\n"


def render_invalid(final_output: str) -> str:
    final_output = (final_output or "").strip()
    st.error(final_output)
    return final_output


def render_structured_output(state: Dict[str, Any]) -> str:
    executive_summary = state.get("executive_summary", "") or ""
    email_to = state.get("email_to", "") or ""
    email_subject = state.get("email_subject", "") or ""
    email_body = state.get("email_body", "") or ""
    email = state.get("email", "") or ""
    actions: List[Dict[str, Any]] = state.get("actions", []) or []
    sources = state.get("sources", "") or ""

    md = ""

    if executive_summary.strip():
        st.markdown("### Executive Summary")
        st.write(executive_summary.strip())
        md += _md_block("Executive Summary", executive_summary)

    st.markdown("### Client-ready Email")
    if email_to or email_subject or email_body:
        st.markdown(f"**To:** {email_to or '—'}")
        st.markdown(f"**Subject:** {email_subject or '—'}")
        st.divider()
        st.write((email_body or "").strip())
        email_md = f"**To:** {email_to or '—'}\n\n**Subject:** {email_subject or '—'}\n\n{(email_body or '').strip()}"
        md += _md_block("Client-ready Email", email_md)
    elif email.strip():
        st.write(email.strip())
        md += _md_block("Client-ready Email", email)

    st.markdown("### Action List")
    if actions:
        st.dataframe(actions, use_container_width=True, hide_index=True)
        md += _md_block("Action List", "\n".join([f"- {a}" for a in actions]))
    else:
        st.write("")

    if sources.strip():
        st.markdown("### Sources & Citations")
        st.write(sources.strip())
        md += _md_block("Sources & Citations", sources)

    return md.strip() if md.strip() else "(No output fields returned.)"


def render_runtime_error(message: str, details: str = "") -> None:
    st.error(message)
    if details:
        with st.expander("Details"):
            st.code(details)
