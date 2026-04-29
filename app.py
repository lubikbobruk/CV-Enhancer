"""Streamlit interface"""

import streamlit as st
from src.parsing import *
from src.retrieval.tfidf_retriever import TfidfRetriever

st.title("CV Enhancer")

left, right = st.columns(2)

with left:
    cv_file = st.file_uploader("CV", type=["pdf", "docx"])

with right:
    job_ad = st.text_area("Job ad", height=300)

if st.button("Enhance"):
    if cv_file is None:
        st.warning("Upload a CV first.")
        st.stop()

    try:
        text = extract_text(cv_file, cv_file.name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not text:
        st.warning("Couldn't extract any text from this file. \
                    If it's a scanned PDF, try a text-based version.")
        st.stop()

    chunks = chunk(text)
    st.info(f"Parsed {len(chunks)} chunks from the CV.")
    with st.expander("Preview chunks"):
        for i, block in enumerate(chunks):
            st.markdown(f"**Chunk {i}**")
            st.text(block)

    if not job_ad.strip():
        st.info("Paste a job ad to see ranked chunks.")
        st.stop()

    retriever = TfidfRetriever()
    retriever.fit(chunks)
    top = retriever.query(job_ad, k=5)

    with st.expander("Top-k chunks for this job ad (TF-IDF)", expanded=True):
        for rank, (idx, score) in enumerate(top, start=1):
            st.markdown(f"**#{rank} — chunk {idx} — score {score:.3f}**")
            st.text(chunks[idx])

