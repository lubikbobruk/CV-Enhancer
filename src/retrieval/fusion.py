"""Reciprocal Rank Fusion implementation."""

# to smooth the weights derived from the ranks
DEFAULT_K_RRF = 60

# Pass lis of all rankings to get one final ranking
def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]],
    k: int,
    k_rrf: int = DEFAULT_K_RRF) -> list[tuple[int, float]]:
  """Fuse ranked lists into a single top-k ranking by RRF."""
  if not rankings:
    raise ValueError("rankings must contain at least one ranked list")

  fused: dict[int, float] = {}
  for ranking in rankings:
    for rank, (chunk_id, _score) in enumerate(ranking, start=1):
      fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (k_rrf + rank)

  ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
  return ordered[:k]
