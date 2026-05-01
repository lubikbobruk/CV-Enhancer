"""Environment loading and project-wide constants."""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.")
