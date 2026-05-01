"""Session-state machine for the three-screen flow.

Stages:
    "upload"  -> user uploads CV and pastes job ad
    "select"  -> chunked + ranked; user picks chunks to rewrite
    "results" -> Gemini rewrites returned; user accepts/rejects + downloads PDF

State is preserved across stage transitions so "Back" doesn't recompute
expensive work (parsing, embedding, Gemini calls). "Start over" wipes
everything via reset().
"""

import streamlit as st


_STAGES = ("upload", "select", "results")


def init() -> None:
    """Initialize session_state keys on first run. Idempotent."""
    defaults = {
        "stage": "upload",
        "cv_text": "",
        "chunks": [],
        "filtered_ad": "",
        "ranking": [],          # list[tuple[chunk_id, rrf_score]] least-relevant first
        "dense_scores": {},     # chunk_id -> cosine in [0, 1]
        "selected_ids": set(),
        "expanded_ids": set(),  # chunks the user expanded inline
        "result": None,         # EnhancementResult | None
        "accepted_ids": set(),  # rewrites the user is applying (default: all)
        "dropped_count": 0,     # rewrites filtered out for non-positive delta
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def goto(stage: str) -> None:
    """Set the current stage and trigger a rerun."""
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
    """Return the active stage."""
    return st.session_state.get("stage", "upload")
