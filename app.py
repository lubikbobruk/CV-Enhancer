"""Streamlit interface"""

import streamlit as st

from src.generation.gemini_client import test_call

st.title("CV Enhancer")

left, right = st.columns(2)

with left:
    cv_file = st.file_uploader("CV", type=["pdf", "docx"])

with right:
    job_ad = st.text_area("Job ad", height=300)

if st.button("Enhance"):
    try:
        reply = test_call("Welcome a person to the CV-enhancer project in enthusiastic manner.")
        st.success(reply)
    except Exception as exc:
        st.error(f"Gemini call failed: {exc}")
