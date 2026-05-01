"""Session-state machine for the upload -> select -> results flow."""

import streamlit as st


_STAGES = ("upload", "select", "results")


def init() -> None:
    """Initialize session_state keys on first run. Idempotent."""
    defaults = {
        "stage": "upload",
        "cv_text": "",
        "chunks": [],
        "filtered_ad": "",
        "ranking": [],
        "dense_scores": {},
        "selected_ids": set(),
        "result": None,
        "accepted_ids": set(),
        "dropped_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def goto(stage: str) -> None:
    if stage not in _STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    st.session_state.stage = stage
    st.rerun()


def reset() -> None:
    """Wipe all session state and return to the upload screen."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init()
    st.rerun()


def current() -> str:
    return st.session_state.get("stage", "upload")
