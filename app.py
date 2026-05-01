"""Streamlit entry point.

Thin dispatcher: initializes session state, injects CSS, renders the
screen for the current stage. All real work lives in src/ui/*.
"""

import streamlit as st

from src.ui import results, select, state, styles, upload


st.set_page_config(
    page_title="CV Enhancer",
    layout="wide",
    initial_sidebar_state="collapsed",
)
state.init()
styles.inject()


with st.sidebar:
    st.markdown("### CV Enhancer")
    st.caption(f"Stage: `{state.current()}`")
    if st.button("Start over", use_container_width=True):
        state.reset()


_RENDERERS = {
    "upload": upload.render,
    "select": select.render,
    "results": results.render,
}

_RENDERERS[state.current()]()
