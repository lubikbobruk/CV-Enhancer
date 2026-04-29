"""Unit tests for the TF-IDF retriever."""

import pytest
from src.retrieval.tfidf_retriever import TfidfRetriever


CHUNKS = [
    "Python developer with five years of Django and Flask experience building REST APIs.",
    "Machine learning engineer skilled in PyTorch, TensorFlow, and computer vision research.",
    "Senior project manager leading agile teams and stakeholder communication across departments.",
]


def test_tfidf_query_matching_chunk_ranks_first() -> None:

    retriever = TfidfRetriever()

    # (chunk_index, score)
    retriever.fit(CHUNKS)

    # ranking
    result = retriever.query("Python Django REST", k=3)

    assert result[0][0] == 0
    assert result[0][1] > result[1][1]


def test_tfidf_ml_query_picks_ml_chunk() -> None:
    retriever = TfidfRetriever()
    retriever.fit(CHUNKS)

    result = retriever.query("deep learning PyTorch vision", k=3)

    assert result[0][0] == 1


def test_tfidf_unseen_query_returns_zero_scores() -> None:
    retriever = TfidfRetriever()
    retriever.fit(CHUNKS)

    result = retriever.query("xylophone quantum unicorn", k=3)

    assert len(result) == 3
    assert all(score == 0.0 for _, score in result)


def test_tfidf_k_larger_than_corpus_returns_all() -> None:
    retriever = TfidfRetriever()
    retriever.fit(CHUNKS)

    result = retriever.query("Python", k=10)

    assert len(result) == 3


def test_tfidf_fit_then_query_returns_indices_and_scores() -> None:
    retriever = TfidfRetriever()
    retriever.fit(CHUNKS)

    result = retriever.query("Python Django", k=3)

    for idx, score in result:
        assert isinstance(idx, int)
        assert isinstance(score, float)
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)
    indices = [idx for idx, _ in result]
    assert all(0 <= idx < 3 for idx in indices)


def test_tfidf_empty_corpus_raises_value_error() -> None:
    retriever = TfidfRetriever()

    with pytest.raises(ValueError):
        retriever.fit([])


def test_tfidf_query_before_fit_raises_runtime_error() -> None:
    retriever = TfidfRetriever()

    with pytest.raises(RuntimeError):
        retriever.query("anything", k=1)


def test_tfidf_duplicate_chunks_receive_equal_scores() -> None:
    duplicate = "Python developer with Django and Flask experience."
    chunks = [duplicate, duplicate, "Project manager leading agile teams across departments."]
    retriever = TfidfRetriever()
    retriever.fit(chunks)

    result = retriever.query("Python Django", k=3)
    score_by_idx = dict(result)

    assert score_by_idx[0] == score_by_idx[1]
    assert score_by_idx[0] > score_by_idx[2]


def test_tfidf_blank_corpus_raises_value_error() -> None:
    retriever = TfidfRetriever()

    # sklearn rejects a corpus with no usable vocabulary.
    with pytest.raises(ValueError):
        retriever.fit([""])
