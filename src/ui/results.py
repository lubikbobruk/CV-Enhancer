"""Results screen: per-chunk rewrites (left) + aggregate delta + PDF (right).

Left pane: one expander per rewrite, showing original vs rewritten,
Gemini's reason, an accept/reject toggle, and that chunk's individual
post-rewrite cosine.

Right pane: the aggregate before -> after similarity card across the full
CV (originals where not accepted, rewrites where accepted), plus an
inline PDF preview and download button.
"""

import base64

import numpy as np
import streamlit as st

from src.output.pdf_writer import build_pdf
from src.retrieval.embedding_retriever import load_minilm
from src.ui import state


def _toggle_accept(chunk_id: int) -> None:
    acc = st.session_state.accepted_ids
    if chunk_id in acc:
        acc.discard(chunk_id)
    else:
        acc.add(chunk_id)


def _final_chunks() -> tuple[list[str], dict[int, str]]:
    """Return (final_chunk_list, overrides_map) honoring accept toggles."""
    chunks = list(st.session_state.chunks)
    result = st.session_state.result
    accepted = st.session_state.accepted_ids
    overrides: dict[int, str] = {}
    for rewrite in result.rewrites:
        if rewrite.chunk_id in accepted:
            chunks[rewrite.chunk_id] = rewrite.rewritten_text
            overrides[rewrite.chunk_id] = rewrite.rewritten_text
    return chunks, overrides


def _mean_cosine(chunks: list[str], job_ad: str) -> float:
    """Mean dense cosine of every chunk against the (filtered) job ad."""
    model = load_minilm()
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    q = model.encode([job_ad], convert_to_numpy=True, normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = embeddings @ q
    return float(np.clip(scores.mean(), 0.0, 1.0))


def _render_left_pane() -> None:
    st.subheader("Rewrites")
    result = st.session_state.result
    chunks = st.session_state.chunks
    accepted = st.session_state.accepted_ids

    if not result.rewrites:
        st.info("Gemini returned no rewrites.")
        return

    model = load_minilm()
    job_ad = st.session_state.filtered_ad
    job_emb = model.encode([job_ad], convert_to_numpy=True, normalize_embeddings=True)
    job_emb = np.asarray(job_emb, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(job_emb)
    if norm > 0:
        job_emb = job_emb / norm

    for rewrite in result.rewrites:
        cid = rewrite.chunk_id
        original = chunks[cid]
        rewritten_emb = model.encode(
            [rewrite.rewritten_text], convert_to_numpy=True, normalize_embeddings=True
        )
        rewritten_emb = np.asarray(rewritten_emb, dtype=np.float32).reshape(-1)
        new_score = float(np.clip(rewritten_emb @ job_emb, 0.0, 1.0))
        old_score = st.session_state.dense_scores.get(cid, 0.0)
        delta = round(new_score * 100) - round(old_score * 100)
        delta_str = f"+{delta}" if delta >= 0 else str(delta)

        title = (
            f"Chunk {cid} · {round(old_score * 100)}% → "
            f"{round(new_score * 100)}% ({delta_str} pts)"
        )
        with st.expander(title, expanded=True):
            st.checkbox(
                "Accept this rewrite",
                key=f"acc_{cid}",
                value=cid in accepted,
                on_change=_toggle_accept,
                args=(cid,),
            )
            st.markdown("**Original**")
            st.text(original)
            st.markdown("**Rewritten**")
            st.text(rewrite.rewritten_text)
            st.markdown("**Reason**")
            st.caption(rewrite.reason)


def _render_right_pane() -> None:
    st.subheader("Overall fit")

    final_chunks, overrides = _final_chunks()
    original_chunks = st.session_state.chunks
    job_ad = st.session_state.filtered_ad

    with st.spinner("Scoring final CV..."):
        before = _mean_cosine(original_chunks, job_ad)
        after = _mean_cosine(final_chunks, job_ad)

    before_pct = round(before * 100)
    after_pct = round(after * 100)
    st.markdown(
        f"""
        <div class="delta-card">
            <span class="delta-before">{before_pct}%</span>
            <span class="delta-arrow">→</span>
            <span class="delta-after">{after_pct}%</span>
            <div class="delta-label">CV-vs-job-ad similarity (mean cosine)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.subheader("Enhanced CV")

    pdf_bytes = build_pdf(original_chunks, overrides=overrides)
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        'width="100%" height="500" style="border: 1px solid #ddd; border-radius: 8px;">'
        "</iframe>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="enhanced_cv.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )


def render() -> None:
    if st.session_state.result is None:
        st.error("No rewrites available. Go back and run Enhance.")
        if st.button("← Back to selection"):
            state.goto("select")
        return

    st.title("Enhanced CV")
    if st.button("← Back to selection"):
        state.goto("select")

    left, right = st.columns(2)
    with left:
        _render_left_pane()
    with right:
        _render_right_pane()
