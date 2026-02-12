import streamlit as st


def apply_page_config() -> None:
    st.set_page_config(
        page_title="Enterprise Multi-Agent Copilot",
        page_icon="🧠",
        layout="centered",
    )


def apply_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 820px;
                padding-top: 1.2rem;
                padding-bottom: 3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
