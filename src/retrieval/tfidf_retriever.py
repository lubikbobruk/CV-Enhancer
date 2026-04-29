"""TF-IDF retriever."""

import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg
from src.retrieval.base import Retriever

# Compute once at compile time
# Take into account only words
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Simulating sklearn's ENGLISH_STOP_WORDS

_STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at
    be because been before being below between both but by
    could
    did do does doing down during
    each
    few for from further
    had has have having he her here hers herself him himself his how
    i if in into is it its itself
    just
    me more most my myself
    no nor not now
    of off on once only or other our ours ourselves out over own
    same she should so some such
    than that the their theirs them themselves then there these they this those through to too
    under until up
    very
    was we were what when where which while who whom why will with would
    you your yours yourself yourselves
    """.split()
)

def _tokenize(text: str) -> list[str]:
  """Returns preprocessed list of terms."""
  matches = _TOKEN_RE.findall(text.lower())
  return [t for t in matches if t not in _STOP_WORDS]


def _ngrams(tokens: list[str]) -> list[str]:
  """Unigrams + bigrams. Bigrams are pairs of unigrams."""
  bigrams = []
  for i in range(len(tokens) - 1):
      bigrams.append(tokens[i] + " " + tokens[i + 1])
  return tokens + bigrams


class TfidfRetriever(Retriever):
  """TF-IDF vector-space retriever."""

  def __init__(self) -> None:
    # .secind is the column's index 
    self._vocab: dict[str, int] = {}
    self._idf: np.ndarray | None = None
    self._matrix: csr_matrix | None = None
    self._corpus_size = 0

  def fit(self, chunks: list[str]) -> None:
      """Build vocabulary, IDF vector, and the L2-normalized TF-IDF matrix."""
      if not chunks:
          raise ValueError("Cannot fit on empty corpus")

      # list of ngrammed chunk terms
      tokenized : list[list[str]] = []
      for c in chunks:
        tokens = _tokenize(c)
        tokens_with_bigrams : list[str] = _ngrams(tokens)
        # append with list of ngrams of specific chunk
        tokenized.append(tokens_with_bigrams)

      unique_terms = set()
      for terms in tokenized:
          unique_terms.update(terms)
      if not unique_terms:
          raise ValueError("Empty vocabulary.")

      sorted_terms = sorted(unique_terms)
      self._vocab = {}
      for j in range(len(sorted_terms)):
        term = sorted_terms[j]
        self._vocab[term] = j
      chunk_amount = len(chunks)
      term_amount = len(self._vocab)

      # Document frequency
      df = np.zeros(term_amount, dtype=np.float64)
      for terms in tokenized:
          for term in set(terms):
              df[self._vocab[term]] += 1

      # IDF = log2(n/df)
      self._idf = np.log2(chunk_amount / df)

      # Max-norm TF + IDF weighting, written into CSR triplets.
      chunk_index: list[int] = []
      term_index: list[int] = []
      weight_value: list[float] = []
      for i, terms in enumerate(tokenized):
          # term frequency
          counts: dict[str, int] = {}
          for t in terms:
              # return the count or 0 if term is not in the dict yet
              counts[t] = counts.get(t, 0) + 1
          if not counts:
              continue
          max_count = max(counts.values())
          for term, frequency in counts.items():
              j = self._vocab[term]
              # max-norm
              tf = frequency / max_count
              weight = tf * self._idf[j]
              if weight:
                  chunk_index.append(i)
                  term_index.append(j)
                  weight_value.append(weight)

      # TF-IDF term-doc matrix
      matrix = csr_matrix((weight_value, (chunk_index, term_index)), shape=(chunk_amount, term_amount), dtype=np.float64)

      # L2-normalize each row to reduce cosine similarity to a dot product
      row_norms = splinalg.norm(matrix, axis=1)

      # Set empty norm to 1 (division safeguard)
      row_norms[row_norms == 0.0] = 1.0

      inverse = 1.0 / row_norms

      # Scale each row by its inverse norm
      matrix = matrix.multiply(inverse[:, None]).tocsr()

      self._matrix = matrix
      self._corpus_size = chunk_amount

  def query(self, text: str, k: int) -> list[tuple[int, float]]:
    """Return top-k (chunk_index, cosine_score), sorted by descending score."""
    if self._matrix is None or self._idf is None:
      raise RuntimeError("Call fit() first")

    terms = _ngrams(_tokenize(text))
    counts: dict[str, int] = {}
    for t in terms:
      if t in self._vocab:
        counts[t] = counts.get(t, 0) + 1

    q = np.zeros(len(self._vocab), dtype=np.float64)

    # if query matches at least one term
    if counts:
      q_max = max(counts.values())
      for term, raw in counts.items():
        j = self._vocab[term]
        q_tf = raw / q_max
        q[j] = q_tf * self._idf[j]

    # L2 normalization
    norm = np.linalg.norm(q)
    # normalize whole vector
    if norm > 0.0:
      q /= norm

    scores = (self._matrix @ q).ravel()

    k = min(k, self._corpus_size)
    
    # emulating descending order (argsort doesn't provide it)
    # return indexes
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]