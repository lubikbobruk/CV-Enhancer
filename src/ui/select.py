"""Select screen: click chunks to move them between left (unselected) and right (to-enhance) panes."""

import numpy as np
import streamlit as st

from src.generation.gemini_client import enhance_chunks
from src.generation.schemas import ChunkRewrite, EnhancementResult
from src.retrieval.embedding_retriever import load_minilm
from src.ui import state


_FIT_BAD = 0.35
_FIT_GOOD = 0.55
_PREVIEW_LINES = 3
_SCROLL_HEIGHT = 720  # pixels

# Set True to bypass Gemini during UI iteration; results screen renders with echoed originals.
_DISABLE_GEMINI = False


def _stub_result(pairs: list[tuple[int, str]]) -> EnhancementResult:
    return EnhancementResult(
        rewrites=[
            ChunkRewrite(
                chunk_id=cid,
                rewritten_text=f"[STUB REWRITE] {text}",
                reason="Stubbed rewrite — Gemini calls disabled for UI testing.",
            )
            for cid, text in pairs
        ]
    )


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg


def _fit_marker(score: float) -> str:
    if score < _FIT_BAD:
        return "⚠ weak"
    if score < _FIT_GOOD:
        return "● neutral"
    return "✓ strong"


def _preview(text: str, lines: int = _PREVIEW_LINES) -> str:
    parts = text.split("\n")
    if len(parts) <= lines:
        return text
    return "\n".join(parts[:lines]) + "\n…"


def _add(chunk_id: int) -> None:
    st.session_state.selected_ids.add(chunk_id)


def _remove(chunk_id: int) -> None:
    st.session_state.selected_ids.discard(chunk_id)


def _chunk_button_label(rank: int, score: float, text: str) -> str:
    pct = round(score * 100)
    marker = _fit_marker(score)
    header = f"#{rank}   ·   {pct}% fit   ·   {marker}"
    return f"{header}\n\n{_preview(text)}"


def _render_chunk(
    rank: int,
    chunk_id: int,
    score: float,
    text: str,
    side: str,
) -> None:
    label = _chunk_button_label(rank, score, text)
    on_click = _add if side == "left" else _remove
    button_kwargs = {
        "key": f"{side}_{chunk_id}",
        "on_click": on_click,
        "args": (chunk_id,),
        "use_container_width": True,
    }
    if side == "right":
        button_kwargs["type"] = "primary"
    st.button(label, **button_kwargs)


def _trigger_enhancement() -> None:
    st.session_state.enhancement_pending = True


def _drop_non_improvements(result: EnhancementResult) -> tuple[EnhancementResult, int]:
    """Drop rewrites whose dense cosine vs. job ad didn't improve. Returns (kept, dropped_count)."""
    if not result.rewrites:
        return result, 0

    model = load_minilm()
    job_ad = st.session_state.filtered_ad
    job_emb = model.encode([job_ad], convert_to_numpy=True, normalize_embeddings=True)
    job_emb = np.asarray(job_emb, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(job_emb)
    if norm > 0:
        job_emb = job_emb / norm

    dense_scores = st.session_state.dense_scores
    kept: list[ChunkRewrite] = []
    for rw in result.rewrites:
        new_emb = model.encode(
            [rw.rewritten_text], convert_to_numpy=True, normalize_embeddings=True
        )
        new_emb = np.asarray(new_emb, dtype=np.float32).reshape(-1)
        new_score = float(np.clip(new_emb @ job_emb, 0.0, 1.0))
        old_score = dense_scores.get(rw.chunk_id, 0.0)
        delta = round(new_score * 100) - round(old_score * 100)
        if delta > 0:
            kept.append(rw)

    dropped = len(result.rewrites) - len(kept)
    return EnhancementResult(rewrites=kept), dropped


def _run_enhancement() -> None:
    chunks = st.session_state.chunks
    selected = sorted(st.session_state.selected_ids)
    pairs = [(cid, chunks[cid]) for cid in selected]

    if _DISABLE_GEMINI:
        result = _stub_result(pairs)
    else:
        try:
            result = enhance_chunks(
                full_cv=st.session_state.cv_text,
                job_ad=st.session_state.filtered_ad,
                selected=pairs,
            )
        except Exception as exc:
            st.session_state.enhancement_pending = False
            if _is_rate_limit(exc):
                st.error(
                    "⚠ Gemini rate limit hit (429). "
                    "Free tier allows ~15 requests/minute — wait a moment and try again."
                )
            else:
                st.error(f"Gemini call failed: {exc}")
            return

    result, dropped = _drop_non_improvements(result)
    st.session_state.result = result
    st.session_state.dropped_count = dropped
    st.session_state.enhancement_pending = False
    state.goto("results")


def _render_left_pane(left_ranking: list[tuple[int, float]]) -> None:
    chunks = st.session_state.chunks
    dense_scores = st.session_state.dense_scores

    st.subheader("CV Chunks")
    with st.container(height=_SCROLL_HEIGHT, border=False):
        if not left_ranking:
            total = len(st.session_state.ranking)
            st.markdown(
                f"""
                <div class="all-selected">
                    <div class="checkmark">✓</div>
                    <h3>All {total} chunks selected</h3>
                    <p>Click <strong>Apply</strong> to enhance them, or <strong>Revert all</strong> to start over.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        for rank, (chunk_id, _rrf) in enumerate(left_ranking, start=1):
            cosine = dense_scores.get(chunk_id, 0.0)
            _render_chunk(rank, chunk_id, cosine, chunks[chunk_id], side="left")


def _render_right_pane(right_ranking: list[tuple[int, float]]) -> None:
    chunks = st.session_state.chunks
    dense_scores = st.session_state.dense_scores
    selected_count = len(st.session_state.selected_ids)
    full_ranking = st.session_state.ranking

    st.subheader("Chunks to enhance")

    btn_a, btn_b = st.columns(2)
    with btn_a:
        if st.button("Select all", key="select_all", use_container_width=True):
            st.session_state.selected_ids = {cid for cid, _ in full_ranking}
            st.rerun()
    with btn_b:
        if st.button("Revert all", key="revert_all", use_container_width=True):
            st.session_state.selected_ids = set()
            st.rerun()

    st.button(
        f"Apply ({selected_count})",
        key="apply",
        type="primary",
        disabled=selected_count == 0,
        use_container_width=True,
        help="Select at least one chunk to apply." if selected_count == 0 else None,
        on_click=_trigger_enhancement,
    )

    with st.container(height=_SCROLL_HEIGHT - 140, border=False):
        if not right_ranking:
            st.caption("Click any chunk on the left to enhance it.")
            return
        for rank, (chunk_id, _rrf) in enumerate(right_ranking, start=1):
            cosine = dense_scores.get(chunk_id, 0.0)
            _render_chunk(rank, chunk_id, cosine, chunks[chunk_id], side="right")


def render() -> None:
    st.title("Select Chunks")

    if st.session_state.get("enhancement_pending"):
        with st.spinner("Rewriting selected chunks with Gemini..."):
            _run_enhancement()
        st.rerun()
        return

    if st.button("← Back to upload"):
        state.goto("upload")

    ranking = st.session_state.ranking
    selected = st.session_state.selected_ids

    left_ranking = [(cid, score) for cid, score in ranking if cid not in selected]
    right_ranking = [(cid, score) for cid, score in ranking if cid in selected]

    left_col, right_col = st.columns([3, 1])
    with left_col:
        _render_left_pane(left_ranking)
    with right_col:
        _render_right_pane(right_ranking)
