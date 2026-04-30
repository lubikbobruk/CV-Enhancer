"""Streamlit interface"""

import streamlit as st
from src.parsing import *
from src.retrieval.embedding_retriever import DenseRetriever, load_minilm
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.tfidf_retriever import TfidfRetriever


def _render_ranking(title: str, ranking: list[tuple[int, float]], chunks: list[str]) -> None:
    st.markdown(f"**{title}**")
    for rank, (idx, score) in enumerate(ranking, start=1):
        st.markdown(f"#{rank} — chunk {idx} — score {score:.3f}")
        st.text(chunks[idx])

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

    tfidf = TfidfRetriever()
    tfidf.fit(chunks)
    tfidf_ranking = tfidf.query(job_ad, k=5)

    with st.spinner("Encoding chunks with MiniLM..."):
        dense = DenseRetriever(load_minilm())
        dense.fit(chunks)
    dense_ranking = dense.query(job_ad, k=5)

    fused_ranking = reciprocal_rank_fusion([tfidf_ranking, dense_ranking], k=5)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        _render_ranking("TF-IDF", tfidf_ranking, chunks)
    with col_b:
        _render_ranking("Dense (MiniLM)", dense_ranking, chunks)
    with col_c:
        _render_ranking("Fused (RRF)", fused_ranking, chunks)

