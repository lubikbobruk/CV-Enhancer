"""Select screen: two-pane move-on-click layout.

Left (3/4 width): "CV Chunks" — every chunk not yet selected, ordered
least-relevant first. Each chunk is itself a full-width button: clicking
it moves the chunk to the right pane.

Right (1/4 width): "Chunks to enhance" — selected chunks, rendered as
type="primary" buttons (green outline via CSS). Clicking removes back to
the left pane. Action buttons sit above the list: Select all, Revert
all, Apply.

Each pane uses st.container(height=...) so the LIST scrolls independently;
the pane title sits outside the scroll container and stays put.
"""

import streamlit as st

from src.generation.gemini_client import enhance_chunks
from src.generation.schemas import ChunkRewrite, EnhancementResult
from src.ui import state


_FIT_BAD = 0.35
_FIT_GOOD = 0.55
_PREVIEW_LINES = 3
_SCROLL_HEIGHT = 720  # pixels

# Toggle to skip live Gemini calls during UI iteration. Returns a canned
# EnhancementResult with the original text echoed back, so the results
# screen renders without burning API quota.
_DISABLE_GEMINI = True


def _stub_result(pairs: list[tuple[int, str]]) -> EnhancementResult:
    """Return a canned EnhancementResult for UI testing."""
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
    """Best-effort detection of a 429 from langchain/google-genai."""
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
    """Compose the multiline label rendered inside a chunk button."""
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
    """Render one chunk as a full-width clickable button."""
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


def _run_enhancement() -> None:
    chunks = st.session_state.chunks
    selected = sorted(st.session_state.selected_ids)
    pairs = [(cid, chunks[cid]) for cid in selected]

    if _DISABLE_GEMINI:
        result = _stub_result(pairs)
    else:
        with st.spinner("Rewriting selected chunks with Gemini..."):
            try:
                result = enhance_chunks(
                    full_cv=st.session_state.cv_text,
                    job_ad=st.session_state.filtered_ad,
                    selected=pairs,
                )
            except Exception as exc:
                if _is_rate_limit(exc):
                    st.error(
                        "⚠ Gemini rate limit hit (429). "
                        "Free tier allows ~15 requests/minute — wait a moment and try again."
                    )
                else:
                    st.error(f"Gemini call failed: {exc}")
                return

    st.session_state.result = result
    st.session_state.accepted_ids = {r.chunk_id for r in result.rewrites}
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

    if st.button(
        f"Apply ({selected_count})",
        key="apply",
        disabled=selected_count == 0,
        use_container_width=True,
        help="Select at least one chunk to apply." if selected_count == 0 else None,
    ):
        _run_enhancement()

    with st.container(height=_SCROLL_HEIGHT - 140, border=False):
        if not right_ranking:
            st.caption("Click any chunk on the left to enhance it.")
            return
        for rank, (chunk_id, _rrf) in enumerate(right_ranking, start=1):
            cosine = dense_scores.get(chunk_id, 0.0)
            _render_chunk(rank, chunk_id, cosine, chunks[chunk_id], side="right")


def render() -> None:
    back_col, _ = st.columns([3, 8])
    with back_col:
        if st.button("← Back to upload", use_container_width=True):
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
