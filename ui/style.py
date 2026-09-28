import html
from pathlib import Path

import streamlit as st


def apply_style() -> None:
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-heading"><div class="eyebrow">{html.escape(eyebrow)}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>",
        unsafe_allow_html=True,
    )


def note(label: str, text: str) -> None:
    st.markdown(
        f'<div class="editorial-note"><div class="eyebrow">{html.escape(label)}</div>'
        f"<p>{html.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )
