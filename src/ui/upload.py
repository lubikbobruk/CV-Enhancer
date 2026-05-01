"""Upload screen: CV file + job ad textarea + Enhance button.

Clicking Enhance runs parsing -> chunking -> filtering -> ranking once,
stores results in session_state, then transitions to the select stage.
"""

import streamlit as st

from src.parsing import chunk_cv, extract_text
from src.parsing.job_ad_filter import filter_role_relevant
from src.retrieval.embedding_retriever import DenseRetriever, load_minilm
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.ui import state


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>CV Enhancer</h1>
            <p>Upload your CV, paste a job ad. The retriever surfaces the chunks
            worth rewriting; Gemini reframes them toward the role without inventing facts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 3, 1])
    with center:
        left, right = st.columns(2)
        with left:
            cv_file = st.file_uploader("CV", type=["pdf", "docx"])
        with right:
            job_ad = st.text_area(
                "Job ad",
                height=240,
                placeholder="Paste the full job advertisement here.",
            )

        if not st.button("Enhance", key="enhance", use_container_width=True):
            return

    if cv_file is None:
        st.warning("Upload a CV first.")
        return
    if not job_ad.strip():
        st.warning("Paste a job ad first.")
        return

    try:
        cv_text = extract_text(cv_file, cv_file.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    if not cv_text.strip():
        st.error("Couldn't extract any text. If it's a scanned PDF, try a text-based version.")
        return

    with st.spinner("Parsing, chunking, and ranking against the job ad..."):
        chunks = chunk_cv(cv_text)
        if not chunks:
            st.error("CV produced no chunks after parsing.")
            return

        model = load_minilm()
        filtered_ad = filter_role_relevant(job_ad, model=model)

        tfidf = TfidfRetriever()
        tfidf.fit(chunks)
        dense = DenseRetriever(model)
        dense.fit(chunks)

        k = len(chunks)
        tfidf_ranking = tfidf.query(filtered_ad, k=k)
        dense_ranking = dense.query(filtered_ad, k=k)
        fused = reciprocal_rank_fusion([tfidf_ranking, dense_ranking], k=k)

        # Least-relevant first per Phase 4 Q2 decision.
        ranking = list(reversed(fused))
        # Per-chunk dense cosine for the % display (Q4b).
        dense_scores = {cid: max(0.0, score) for cid, score in dense_ranking}

    st.session_state.cv_text = cv_text
    st.session_state.chunks = chunks
    st.session_state.filtered_ad = filtered_ad
    st.session_state.ranking = ranking
    st.session_state.dense_scores = dense_scores
    st.session_state.selected_ids = set()
    st.session_state.expanded_ids = set()
    st.session_state.result = None
    st.session_state.accepted_ids = set()

    state.goto("select")
