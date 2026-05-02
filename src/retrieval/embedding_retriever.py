"""Dense retriever backed by sentence-transformers (MiniLM)."""

from functools import lru_cache
import numpy as np
from src.retrieval.base import Retriever

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# singleton 
@lru_cache(maxsize=1)
def load_minilm():
  """Load MiniLM once per process."""
  from sentence_transformers import SentenceTransformer
  return SentenceTransformer(_DEFAULT_MODEL_NAME)


class DenseRetriever(Retriever):
  """Sentence-embedding retriever. Cosine similarity via dot product on L2-normalized vectors."""

  def __init__(self, model) -> None:
    self._model = model
    self._matrix: np.ndarray | None = None
    self._corpus_size = 0

  def fit(self, chunks: list[str]) -> None:
    """Encode the corpus and store L2-normalized embeddings."""
    if not chunks:
      raise ValueError("Cannot fit on empty corpus")

    embeddings = self._model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Defence-in-depth: re-normalize in case the encoder ignored the flag
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0.0] = 1.0
    embeddings = embeddings/norms

    self._matrix = embeddings
    self._corpus_size = embeddings.shape[0]

  def query(self, text: str, k: int) -> list[tuple[int, float]]:
    """Return top-k sorted by descending score."""
    if self._matrix is None:
      raise RuntimeError("Call fit() first.")

    q = self._model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32).reshape(-1)

    norm = np.linalg.norm(q)
    if norm > 0.0:
      q = q / norm

    # Cosine = dot product after L2 normalization
    scores = self._matrix @ q

    k = min(k, self._corpus_size)
    top = np.argsort(-scores)[:k]

    # chunk index and cosine score
    return [(int(i), float(scores[i])) for i in top]
