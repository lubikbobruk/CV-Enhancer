"""Unit tests for Reciprocal Rank Fusion."""

import pytest

from src.retrieval.fusion import DEFAULT_K_RRF, reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
  ranking = [(2, 0.9), (5, 0.7), (1, 0.3)]
  result = reciprocal_rank_fusion([ranking], k=3)
  assert [chunk_id for chunk_id, _ in result] == [2, 5, 1]


def test_two_rankings_with_consensus_top_item() -> None:
  tfidf = [(7, 0.9), (3, 0.5), (1, 0.1)]
  dense = [(7, 0.8), (1, 0.6), (3, 0.2)]
  result = reciprocal_rank_fusion([tfidf, dense], k=3)
  assert result[0][0] == 7


def test_disjoint_rankings_keep_all_items() -> None:
  a = [(1, 0.9), (2, 0.5)]
  b = [(3, 0.9), (4, 0.5)]
  result = reciprocal_rank_fusion([a, b], k=4)
  assert {chunk_id for chunk_id, _ in result} == {1, 2, 3, 4}


def test_consensus_beats_a_single_first_place() -> None:
  a = [(9, 0.99), (5, 0.5)]
  b = [(8, 0.99), (5, 0.5)]
  result = reciprocal_rank_fusion([a, b], k=3)
  assert result[0][0] == 5


def test_top_k_truncates_result() -> None:
  a = [(1, 0.9), (2, 0.7), (3, 0.5), (4, 0.3)]
  result = reciprocal_rank_fusion([a], k=2)
  assert len(result) == 2


def test_scores_are_descending() -> None:
  a = [(1, 0.9), (2, 0.7), (3, 0.5)]
  b = [(3, 0.9), (1, 0.7), (2, 0.5)]
  result = reciprocal_rank_fusion([a, b], k=3)
  scores = [s for _, s in result]
  assert scores == sorted(scores, reverse=True)


def test_empty_rankings_raises_value_error() -> None:
  with pytest.raises(ValueError):
    reciprocal_rank_fusion([], k=3)


def test_all_empty_rankings_returns_empty_list() -> None:
  result = reciprocal_rank_fusion([[], []], k=3)
  assert result == []


def test_k_rrf_parameter_changes_score_but_not_order() -> None:
  a = [(1, 0.9), (2, 0.5)]
  b = [(2, 0.9), (1, 0.5)]
  default = reciprocal_rank_fusion([a, b], k=2)
  custom = reciprocal_rank_fusion([a, b], k=2, k_rrf=10)
  assert [c for c, _ in default] == [c for c, _ in custom]
  assert default[0][1] != custom[0][1]
