"""Pydantic schemas for structured Gemini output."""

from pydantic import BaseModel, Field


class ChunkRewrite(BaseModel):
    """One rewritten CV chunk plus the evidence-grounded reason for the edit."""

    chunk_id: int = Field(description="Index of the chunk in the original CV chunk list.")
    rewritten_text: str = Field(description="The rewritten chunk text.")
    reason: str = Field(
        description=(
            "One-sentence justification citing specific evidence from the original "
            "chunk and the job ad. Must ground the edit in concrete content, not "
            "generic claims.")
    )


class EnhancementResult(BaseModel):
    """Batch result for one Enhance-selected-chunks request."""

    rewrites: list[ChunkRewrite] = Field(description="One ChunkRewrite per selected chunk, in the order they were submitted.")
