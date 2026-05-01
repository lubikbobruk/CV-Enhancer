"""Filter a job ad to keep only role/skills content; drop benefits/culture/contact."""

import re
import numpy as np

_MIN_BLOCK_CHARS = 40

_HARD_DROP_RE = re.compile(
    r"@[\w.-]+\.\w+"           # email
    r"|linkedin\.com/in/"      # recruiter LinkedIn
    r"|\+\d[\d\s\-]{7,}"       # phone numbers
    r"|\bapply now\b",
    re.IGNORECASE,
)


# An anchor is a reference example of what a category looks like, written as a sentence and then encoded into a vector.

_ROLE_ANCHORS = [
    "Required technical skills, programming languages, and engineering responsibilities for this position.",
    "Years of experience, frameworks, tools, and qualifications required to perform the job.",
    "What the candidate will build, ship, and contribute to as part of the engineering work."]

_BENEFITS_ANCHORS = [
    "Company benefits, perks, vacation days, health insurance, and cafeteria allowance.",
    "Workplace culture, team events, sports activities, and well-being programs.",
    "Recruiter contact information, application instructions, and the hiring process."]


# use to smooth out the bias of using only one anchor
def _centroid(model, anchors: list[str]) -> np.ndarray:
    """Encode anchors and average to a single L2-normalized centroid."""
    embeddings = model.encode(anchors, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def filter_role_relevant(text: str, model=None) -> str:
    """Return job-ad text without benefits/culture/contact chunks."""
    if model is None:
        return text

    blocks = [b.strip() for b in text.split("\n\n")]
    blocks = [b for b in blocks if len(b) >= _MIN_BLOCK_CHARS]
    if not blocks:
        return text

    role_centroid = _centroid(model, _ROLE_ANCHORS)
    benefits_centroid = _centroid(model, _BENEFITS_ANCHORS)

    block_embeddings = model.encode(blocks, convert_to_numpy=True, normalize_embeddings=True)
    block_embeddings = np.asarray(block_embeddings, dtype=np.float32)

    # Cosine sim == dot product after l2 normalization
    role_scores = block_embeddings @ role_centroid
    benefits_scores = block_embeddings @ benefits_centroid

    kept = [
        b for b, role, bene in zip(blocks, role_scores, benefits_scores)
        if role >= bene and not _HARD_DROP_RE.search(b)
    ]
    if not kept:
        return text
    return "\n\n".join(kept)
