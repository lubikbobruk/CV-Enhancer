"""Custom CSS injection. Selectors are pinned to Streamlit's current DOM — revalidate if the pin moves."""

import streamlit as st


_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1600px;
}

/* Default buttons: purple, locked across hover/focus/active. */
div.stButton > button,
div.stButton > button:hover,
div.stButton > button:focus,
div.stButton > button:focus:not(:active),
div.stButton > button:active {
    background-color: #7e5ab8 !important;
    color: #ffffff !important;
    border: 1px solid #9d7cd8 !important;
    border-radius: 8px;
    font-weight: 600;
    box-shadow: none !important;
    transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
}

/* Apply button (kind="primary" outside the scroll container): green.
   Selected-chunk green is scoped under stVerticalBlockBorderWrapper, so this top-level rule only matches Apply. */
div.stButton > button[kind="primary"],
div.stButton > button[kind="primary"]:focus,
div.stButton > button[kind="primary"]:focus:not(:active) {
    background-color: #2d6a3a !important;
    color: #ffffff !important;
    border: 1px solid #66bb6a !important;
    border-radius: 8px;
    font-weight: 600;
    box-shadow: none !important;
    transition: background-color 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #357a44 !important;
    border-color: #7fd283 !important;
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:active {
    background-color: #245530 !important;
    border-color: #66bb6a !important;
    color: #ffffff !important;
    transform: scale(0.99);
}
div.stButton > button > div,
div.stButton > button > div > p,
div.stButton > button:hover > div,
div.stButton > button:hover > div > p,
div.stButton > button:active > div,
div.stButton > button:active > div > p {
    color: #ffffff !important;
    background-color: transparent !important;
}
div.stButton > button:disabled,
div.stButton > button:disabled:hover {
    background-color: #1a1a24 !important;
    color: #555 !important;
    border-color: #2a2a36 !important;
}
div.stButton > button:disabled > div,
div.stButton > button:disabled > div > p {
    color: #555 !important;
}

/* Chunk buttons (LEFT pane, no kind="primary"): dark, left-aligned. */
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]) {
    width: 100%;
    text-align: left !important;
    background-color: #1a1a24 !important;
    color: #e8e0f0 !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    white-space: pre-wrap !important;
    line-height: 1.45;
    font-weight: 400 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]) > div,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]) > div > p {
    color: #e8e0f0 !important;
    text-align: left !important;
}
/* Left-pane chunk hover/active: purple flash. Selected (kind="primary") chunks keep their static green look. */
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):hover {
    background-color: #22222e !important;
    border-color: #9d7cd8 !important;
    color: #c084fc !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):hover > div,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):hover > div > p {
    color: #c084fc !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):active {
    background-color: #3a2d52 !important;
    border-color: #c084fc !important;
    color: #ffffff !important;
    transform: scale(0.99);
    transition: transform 0.08s ease, background-color 0.08s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):active > div,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:not([kind="primary"]):active > div > p {
    color: #ffffff !important;
}

/* Selected chunk button (kind="primary" inside scroll container): green. */
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"],
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:hover,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:focus,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:active {
    width: 100%;
    text-align: left !important;
    background-color: #11231a !important;
    border: 2px solid #66bb6a !important;
    color: #e8e0f0 !important;
    font-weight: 400 !important;
    padding: 0.9rem 1rem;
    white-space: pre-wrap !important;
    line-height: 1.45;
    transform: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:hover {
    background-color: #163022 !important;
    border-color: #7fd283 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"] > div,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"] > div > p,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:hover > div,
div[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button[kind="primary"]:hover > div > p {
    color: #e8e0f0 !important;
    text-align: left !important;
}

/* All-selected confirmation panel (left pane when empty). */
.all-selected {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 600px;
    padding: 2rem;
    border: 1px dashed #66bb6a;
    border-radius: 12px;
    background: #11231a;
    text-align: center;
}
.all-selected .checkmark {
    font-size: 4rem;
    color: #66bb6a;
    margin-bottom: 1rem;
}
.all-selected h3 {
    color: #e8e0f0;
    margin: 0 0 0.5rem 0;
}
.all-selected p {
    color: #b8a8d0;
    margin: 0;
    max-width: 320px;
}

/* Aggregate delta on the results screen. */
.delta-card {
    text-align: center;
    padding: 2rem 1rem;
    border: 1px solid #2a2a36;
    border-radius: 12px;
    background: #1a1a24;
}
.delta-before { font-size: 1.3rem; color: #aaa; }
.delta-arrow { font-size: 2rem; margin: 0 0.5rem; color: #aaa; }
.delta-after { font-size: 2.4rem; font-weight: 700; color: #66bb6a; }
.delta-label { color: #aaa; margin-top: 0.5rem; font-size: 0.9rem; }

/* Hero on the upload screen. */
.hero {
    text-align: center;
    padding: 1.5rem 1rem 2rem 1rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #2a2a36;
}
.hero h1 {
    font-size: 2.4rem;
    margin: 0;
    background: linear-gradient(90deg, #c084fc, #9d7cd8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #b8a8d0;
    font-size: 1.05rem;
    margin: 0.6rem auto 0 auto;
    max-width: 640px;
}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
