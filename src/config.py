"""Environment loading and project-wide constants."""

import os
from dotenv import load_dotenv

load_dotenv()


def _load_gemini_key() -> str | None:
    """Local environment first, Streamlit Cloud secrets as fallback."""
    if key := os.getenv("GEMINI_API_KEY"):
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


GEMINI_API_KEY = _load_gemini_key()
MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
