"""Results screen: per-chunk rewrites (left), aggregate delta and PDF (right)."""

import base64

import numpy as np
import streamlit as st

from src.output.pdf_writer import build_pdf
from src.retrieval.embedding_retriever import load_minilm
from src.ui import state


def _toggle_active(chunk_id: int) -> None:
    """Accordion toggle: clicking the active row collapses it, otherwise opens it."""
    if st.session_state.get("active_rewrite") == chunk_id:
        st.session_state.active_rewrite = None
    else:
        st.session_state.active_rewrite = chunk_id


def _toggle_accept(chunk_id: int) -> None:
    acc = st.session_state.accepted_ids
    if chunk_id in acc:
        acc.discard(chunk_id)
    else:
        acc.add(chunk_id)
    st.session_state._derived_dirty = True


def _ensure_results_cache() -> None:
    """Encode originals, job ad, and rewrites once per result. Toggling Accept doesn't re-embed."""
    cache_key = id(st.session_state.result)
    if st.session_state.get("_results_cache_key") == cache_key:
        return

    result = st.session_state.result
    original_chunks = list(st.session_state.chunks)
    job_ad = st.session_state.filtered_ad

    model = load_minilm()
    job_emb = model.encode([job_ad], convert_to_numpy=True, normalize_embeddings=True)
    job_emb = np.asarray(job_emb, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(job_emb)
    if n > 0:
        job_emb = job_emb / n

    original_embs = model.encode(
        original_chunks, convert_to_numpy=True, normalize_embeddings=True
    )
    original_embs = np.asarray(original_embs, dtype=np.float32)

    per_rewrite_score: dict[int, float] = {}
    rewrite_emb_by_cid: dict[int, np.ndarray] = {}
    if result.rewrites:
        rewritten_embs = model.encode(
            [rw.rewritten_text for rw in result.rewrites],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        rewritten_embs = np.asarray(rewritten_embs, dtype=np.float32)
        scores = rewritten_embs @ job_emb
        for rw, s, emb in zip(result.rewrites, scores, rewritten_embs):
            per_rewrite_score[rw.chunk_id] = float(np.clip(s, 0.0, 1.0))
            rewrite_emb_by_cid[rw.chunk_id] = emb

    before = (
        float(np.clip((original_embs @ job_emb).mean(), 0.0, 1.0))
        if len(original_embs)
        else 0.0
    )

    st.session_state._results_cache_key = cache_key
    st.session_state._cached_per_rewrite = per_rewrite_score
    st.session_state._cached_original_embs = original_embs
    st.session_state._cached_rewrite_embs = rewrite_emb_by_cid
    st.session_state._cached_job_emb = job_emb
    st.session_state._cached_before = before
    st.session_state.accepted_ids = {rw.chunk_id for rw in result.rewrites}
    st.session_state._derived_dirty = True
    if "active_rewrite" not in st.session_state and result.rewrites:
        st.session_state.active_rewrite = result.rewrites[0].chunk_id


def _refresh_derived() -> None:
    """Recompute after-score and PDF from cached embeddings only — no model.encode()."""
    if not st.session_state.get("_derived_dirty"):
        return

    result = st.session_state.result
    accepted = st.session_state.accepted_ids
    original_chunks = list(st.session_state.chunks)
    original_embs = st.session_state._cached_original_embs
    rewrite_embs = st.session_state._cached_rewrite_embs
    job_emb = st.session_state._cached_job_emb

    final_embs = original_embs.copy() if len(original_embs) else original_embs
    overrides: dict[int, str] = {}
    for rw in result.rewrites:
        if rw.chunk_id in accepted:
            final_embs[rw.chunk_id] = rewrite_embs[rw.chunk_id]
            overrides[rw.chunk_id] = rw.rewritten_text

    after = (
        float(np.clip((final_embs @ job_emb).mean(), 0.0, 1.0))
        if len(final_embs)
        else 0.0
    )
    pdf_bytes = build_pdf(original_chunks, overrides=overrides)

    st.session_state._cached_after = after
    st.session_state._cached_pdf = pdf_bytes
    st.session_state._derived_dirty = False


def _render_left_pane() -> None:
    st.subheader("Rewrites")
    result = st.session_state.result
    chunks = st.session_state.chunks
    dropped = st.session_state.get("dropped_count", 0)

    if dropped:
        plural = "s" if dropped != 1 else ""
        st.caption(
            f"⚠ {dropped} rewrite{plural} didn't improve job-ad fit and "
            f"{'were' if dropped != 1 else 'was'} dropped."
        )

    if not result.rewrites:
        st.info("No rewrites improved job-ad fit. Try selecting different chunks.")
        return

    per_rewrite = st.session_state._cached_per_rewrite
    accepted = st.session_state.accepted_ids
    active = st.session_state.active_rewrite

    for rewrite in result.rewrites:
        cid = rewrite.chunk_id
        new_score = per_rewrite[cid]
        old_score = st.session_state.dense_scores.get(cid, 0.0)
        delta = round(new_score * 100) - round(old_score * 100)
        delta_str = f"+{delta}" if delta >= 0 else str(delta)

        is_open = cid == active
        is_applied = cid in accepted
        arrow = "▾" if is_open else "▸"
        applied_tag = "" if is_applied else "  · skipped"
        title = (
            f"{arrow}  Chunk {cid} · {round(old_score * 100)}% → "
            f"{round(new_score * 100)}% ({delta_str} pts){applied_tag}"
        )
        st.button(
            title,
            key=f"hdr_{cid}",
            on_click=_toggle_active,
            args=(cid,),
            use_container_width=True,
        )
        if is_open:
            st.checkbox(
                "Apply this rewrite to the final CV",
                key=f"acc_{cid}",
                value=is_applied,
                on_change=_toggle_accept,
                args=(cid,),
            )
            st.markdown("**Original**")
            st.text(chunks[cid])
            st.markdown("**Rewritten**")
            st.text(rewrite.rewritten_text)
            st.markdown("**Reason**")
            st.caption(rewrite.reason)


def _render_right_pane() -> None:
    st.subheader("Overall fit")

    before_pct = round(st.session_state._cached_before * 100)
    after_pct = round(st.session_state._cached_after * 100)
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

    pdf_bytes = st.session_state._cached_pdf
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

    _ensure_results_cache()
    _refresh_derived()

    left, right = st.columns(2)
    with left:
        _render_left_pane()
    with right:
        _render_right_pane()
