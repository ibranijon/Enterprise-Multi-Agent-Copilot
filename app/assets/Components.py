import streamlit as st
import pandas as pd
from typing import Any, Dict, List


def render_header() -> None:
    st.markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin-bottom:0.25rem;">Heart Failure Readmission Copilot</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intro() -> None:
    st.markdown(
        """
        <div style="text-align:center; margin-top:0.5rem; margin-bottom:1.25rem;">
            <div style="font-size:1.02rem; line-height:1.6; opacity:0.9;">
                Ask questions using a curated corpus of peer-reviewed research on
                <b>heart failure readmissions</b>, <b>risk stratification</b>, and
                <b>transitional care interventions</b>.
            </div>
            <div style="font-size:0.95rem; line-height:1.6; opacity:0.75; margin-top:0.5rem;">
                You’ll get structured outputs (e.g., executive summary, client-ready email, action list)
                grounded in the documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _md_block(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"### {title}\n\n{body}\n"


def render_invalid(final_output: str) -> str:
    final_output = (final_output or "").strip()
    st.error(final_output)
    return final_output


def _render_actions_friendly(actions: List[Dict[str, Any]]) -> str:
    st.markdown("### Action List")

    if not actions:
        st.write("")
        return ""

    rows = []
    for i, a in enumerate(actions, start=1):
        rows.append(
            {
                "#": i,
                "Task": a.get("task", ""),
                "Owner": a.get("owner", ""),
                "Due date": a.get("due_date", ""),
                "Confidence": (a.get("confidence", "") or "").capitalize(),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


    md = "### Action List\n\n"
    for r in rows:
        md += f"- **{r['Task']}** (Owner: {r['Owner']}, Due: {r['Due date']}, Confidence: {r['Confidence']})\n"
    return md.strip()


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
        email_md = (
            f"**To:** {email_to or '—'}\n\n"
            f"**Subject:** {email_subject or '—'}\n\n"
            f"{(email_body or '').strip()}"
        )
        md += _md_block("Client-ready Email", email_md)
    elif email.strip():
        st.write(email.strip())
        md += _md_block("Client-ready Email", email)

    actions_md = _render_actions_friendly(actions)
    if actions_md:
        md += "\n" + actions_md + "\n"

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
