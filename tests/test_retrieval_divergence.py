"""Smoke test: TF-IDF and dense retrievers must rank differently on a paraphrased query."""

import pytest

from src.retrieval.embedding_retriever import DenseRetriever, load_minilm
from src.retrieval.tfidf_retriever import TfidfRetriever


CHUNKS = [
    "Python developer with five years of Django and Flask experience building REST APIs.",
    "Machine learning engineer skilled in PyTorch, TensorFlow, and computer vision research.",
    "Senior project manager leading agile teams and stakeholder communication across departments.",
]


@pytest.fixture(scope="module")
def tfidf() -> TfidfRetriever:
  retriever = TfidfRetriever()
  retriever.fit(CHUNKS)
  return retriever


@pytest.fixture(scope="module")
def dense() -> DenseRetriever:
  retriever = DenseRetriever(load_minilm())
  retriever.fit(CHUNKS)
  return retriever


def test_paraphrased_query_diverges_between_retrievers(
    tfidf: TfidfRetriever, dense: DenseRetriever) -> None:
  query = "backend web engineer for HTTP services"

  tfidf_top = tfidf.query(query, k=1)[0][0]
  dense_top = dense.query(query, k=1)[0][0]

  assert dense_top == 0
  assert tfidf_top != dense_top
