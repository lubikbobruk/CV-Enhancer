"""Unit tests for the dense (sentence-transformers) retriever."""

import pytest
from src.retrieval.embedding_retriever import DenseRetriever, load_minilm


PYTHON_CHUNK = "Python developer with five years of Django and Flask experience building REST APIs."
ML_CHUNK = "Machine learning engineer skilled in PyTorch, TensorFlow, and computer vision research."
PM_CHUNK = "Senior project manager leading agile teams and stakeholder communication across departments."

CHUNKS = [PYTHON_CHUNK, ML_CHUNK, PM_CHUNK]


@pytest.fixture(scope="module")
def real_retriever() -> DenseRetriever:
  retriever = DenseRetriever(load_minilm())
  retriever.fit(CHUNKS)
  return retriever


def test_python_query_picks_python_chunk(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("looking for a Python web developer", k=3)
  assert result[0][0] == 0


def test_ml_query_picks_ml_chunk(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("hiring for deep learning research", k=3)
  assert result[0][0] == 1


def test_management_query_picks_pm_chunk(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("seeking a leader for cross-functional teams", k=3)
  assert result[0][0] == 2


def test_paraphrased_query_still_picks_python_chunk(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("backend web engineer for HTTP services", k=3)
  assert result[0][0] == 0


def test_scores_are_descending(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("Python developer", k=3)
  scores = [s for _, s in result]
  assert scores == sorted(scores, reverse=True)


def test_k_larger_than_corpus_returns_all(real_retriever: DenseRetriever) -> None:
  result = real_retriever.query("anything", k=99)
  assert len(result) == len(CHUNKS)
