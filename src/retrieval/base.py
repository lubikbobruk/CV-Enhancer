"""Retriever interface shared by classical (TF-IDF) and dense backends."""

from abc import ABC, abstractmethod


class Retriever(ABC):
    """Contract for chunk retrievers used for TF-IDF and sentence-transformers criterias."""

    @abstractmethod
    def fit(self, chunks: list[str]) -> None:
        """Index the corpus. Called once after parsing/chunking."""

    @abstractmethod
    def query(self, text: str, k: int) -> list[tuple[int, float]]:
        """Return top-k (chunk_index, score) sorted by descending score."""
