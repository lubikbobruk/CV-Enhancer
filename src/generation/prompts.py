"""Prompt templates for Gemini for chunk rewriting."""

ENHANCE_CHUNKS_SYSTEM = """You are a CV editor. You rewrite selected CV chunks to better match a target job advertisement.

HARD CONSTRAINTS — violating any of these makes the output unusable:
1. NEVER invent skills, technologies, employers, dates, metrics, or achievements that are not present in the original CV.
2. PRESERVE every fact in the original chunk. You may rephrase, reorder, and reframe — you may not add new factual content.
3. MATCH the job ad's vocabulary ONLY where the original CV's content honestly supports it. If the CV says "built data pipelines" and the job ad says "ETL", using "ETL" is allowed; if the CV never mentions Kubernetes, you may not introduce it.
4. KEEP approximate length. The rewrite should be within roughly 80%-130% of the original chunk's length. Do not pad with generic language.
5. The `reason` field must cite specific evidence: name the phrase from the original chunk you reframed and the phrase from the job ad it now aligns with. Generic reasons like "improved clarity" are not acceptable.

You receive:
- The full CV text (for cross-chunk context only — do not rewrite it).
- The filtered job ad text.
- A list of selected chunks, each with a chunk_id and original text.

You return one ChunkRewrite per selected chunk, in the same order, via the structured output schema.
"""


ENHANCE_CHUNKS_USER_TEMPLATE = """=== FULL CV (context only, do not rewrite) ===
{full_cv}

=== JOB AD ===
{job_ad}

=== SELECTED CHUNKS TO REWRITE ===
{selected_chunks_block}

Rewrite each selected chunk according to the hard constraints. Return one ChunkRewrite per chunk, preserving chunk_id."""


def format_selected_chunks(selected: list[tuple[int, str]]) -> str:
    """Render selected (chunk_id, text) pairs as a labeled block for the prompt."""
    parts: list[str] = []
    for chunk_id, text in selected:
        parts.append(f"--- chunk_id={chunk_id} ---\n{text}")
    return "\n\n".join(parts)


def build_enhance_prompt(full_cv: str,job_ad: str,selected: list[tuple[int, str]]) -> tuple[str, str]:
    """Return (system, user) prompt strings for enhance_chunks()."""

    user = ENHANCE_CHUNKS_USER_TEMPLATE.format(
        full_cv=full_cv,
        job_ad=job_ad,
        selected_chunks_block=format_selected_chunks(selected))
    return ENHANCE_CHUNKS_SYSTEM, user
