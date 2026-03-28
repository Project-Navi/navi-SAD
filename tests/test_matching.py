"""Tests for greedy nearest-neighbor length matching.

Proves: deterministic matching, tie-breaking by smallest index,
iteration order, SubsetSpec output, diagnostics, pair list.
"""

from __future__ import annotations

import pytest

from navi_sad.analysis.matching import match_by_token_count
from navi_sad.analysis.types import MatchingDiagnostics, SubsetSpec


class TestMatchByTokenCount:
    def test_exact_match(self) -> None:
        """Exact token count match: all incorrect matched."""
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        token_counts = {1: 100, 2: 200, 3: 100, 4: 200}
        spec, diag, pairs = match_by_token_count(labels, token_counts)
        assert isinstance(spec, SubsetSpec)
        assert isinstance(diag, MatchingDiagnostics)
        assert spec.n_correct == 2
        assert spec.n_incorrect == 2
        assert spec.included_indices == frozenset({1, 2, 3, 4})
        assert len(pairs) == 2

    def test_nearest_neighbor(self) -> None:
        """Non-exact: match to closest correct sample."""
        labels = {1: "correct", 2: "correct", 3: "incorrect"}
        token_counts = {1: 100, 2: 200, 3: 110}
        spec, _diag, pairs = match_by_token_count(labels, token_counts)
        # incorrect=3 (110 tokens) matches correct=1 (100 tokens, gap=10)
        assert 3 in spec.included_indices
        assert 1 in spec.included_indices
        assert pairs == [(1, 3)]

    def test_without_replacement(self) -> None:
        """Once a correct sample is matched, it's consumed."""
        labels = {1: "correct", 2: "incorrect", 3: "incorrect"}
        token_counts = {1: 100, 2: 100, 3: 100}
        spec, diag, pairs = match_by_token_count(labels, token_counts)
        # Only 1 correct available -> only 1 pair
        assert spec.n_correct == 1
        assert spec.n_incorrect == 1
        assert diag.n_incorrect_dropped == 1
        assert len(pairs) == 1

    def test_tie_broken_by_smallest_index(self) -> None:
        """Two correct at same distance -> smallest dataset_index wins."""
        labels = {1: "correct", 3: "correct", 2: "incorrect"}
        token_counts = {1: 100, 3: 100, 2: 100}
        _spec, _diag, pairs = match_by_token_count(labels, token_counts)
        # Both correct (1, 3) at distance 0 from incorrect 2.
        # Tie broken by smallest correct index -> 1
        assert pairs == [(1, 2)]

    def test_iteration_order_ascending_incorrect(self) -> None:
        """Incorrect samples iterated in ascending dataset_index order."""
        labels = {1: "correct", 2: "correct", 10: "incorrect", 5: "incorrect"}
        # incorrect 5 goes first (ascending), 10 second
        token_counts = {1: 100, 2: 200, 10: 195, 5: 105}
        spec, _diag, pairs = match_by_token_count(labels, token_counts)
        # incorrect 5 (105) matches correct 1 (100, gap=5) -> pair (1, 5)
        # incorrect 10 (195) matches remaining correct 2 (200, gap=5) -> pair (2, 10)
        assert spec.included_indices == frozenset({1, 2, 5, 10})
        assert pairs == [(1, 5), (2, 10)]

    def test_diagnostics_counts(self) -> None:
        """Diagnostics report correct before/after/dropped counts."""
        labels = {1: "correct", 2: "correct", 3: "correct", 4: "incorrect"}
        token_counts = {1: 100, 2: 200, 3: 300, 4: 150}
        _spec, diag, _pairs = match_by_token_count(labels, token_counts)
        assert diag.n_correct_before == 3
        assert diag.n_incorrect_before == 1
        assert diag.n_correct_after == 1
        assert diag.n_incorrect_after == 1
        assert diag.n_correct_dropped == 2
        assert diag.n_incorrect_dropped == 0

    def test_diagnostics_token_stats(self) -> None:
        """Diagnostics report mean token counts."""
        labels = {1: "correct", 2: "correct", 3: "incorrect"}
        token_counts = {1: 100, 2: 200, 3: 110}
        _spec, diag, _pairs = match_by_token_count(labels, token_counts)
        assert diag.mean_tokens_correct_before == pytest.approx(150.0)
        assert diag.mean_tokens_incorrect_before == pytest.approx(110.0)
        # Matched: correct=1 (100), incorrect=3 (110)
        assert diag.mean_tokens_correct_after == pytest.approx(100.0)
        assert diag.mean_tokens_incorrect_after == pytest.approx(110.0)
        assert diag.max_pair_token_gap == 10
        assert diag.mean_pair_token_gap == pytest.approx(10.0)

    def test_provenance_name(self) -> None:
        labels = {1: "correct", 2: "incorrect"}
        token_counts = {1: 100, 2: 100}
        spec, _diag, _pairs = match_by_token_count(labels, token_counts)
        assert spec.provenance_name == "length_matched"

    def test_no_incorrect_returns_empty(self) -> None:
        """No incorrect samples -> empty match, no pairs."""
        labels = {1: "correct", 2: "correct"}
        token_counts = {1: 100, 2: 200}
        spec, _diag, pairs = match_by_token_count(labels, token_counts)
        assert spec.n_correct == 0
        assert spec.n_incorrect == 0
        assert len(spec.included_indices) == 0
        assert pairs == []

    def test_pairs_are_correct_incorrect(self) -> None:
        """Each pair is (correct_idx, incorrect_idx)."""
        labels = {1: "correct", 2: "incorrect", 3: "correct", 4: "incorrect"}
        token_counts = {1: 100, 2: 100, 3: 200, 4: 200}
        _spec, _diag, pairs = match_by_token_count(labels, token_counts)
        for correct_idx, incorrect_idx in pairs:
            assert labels[correct_idx] == "correct"
            assert labels[incorrect_idx] == "incorrect"
