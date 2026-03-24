"""Tests for ordinal pattern extraction and permutation entropy."""

import json
import math
from itertools import permutations
from pathlib import Path

from navi_sad.signal.ordinal import (
    extract_ordinal_patterns,
    permutation_entropy,
    permutation_to_index,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ordinal_cases.json"
with open(_FIXTURE_PATH) as _f:
    _CASES = json.load(_f)


# ===========================================================================
# TestPermutationToIndex
# ===========================================================================
class TestPermutationToIndex:
    def test_identity_is_zero(self) -> None:
        """[0, 1, 2] (identity permutation) should map to index 0."""
        assert permutation_to_index([0, 1, 2], 3) == 0

    def test_reverse_is_max(self) -> None:
        """[2, 1, 0] (reverse permutation) should map to index D! - 1 = 5."""
        assert permutation_to_index([2, 1, 0], 3) == 5

    def test_d4_all_unique(self) -> None:
        """All 24 permutations of range(4) should produce 24 unique indices in [0, 23]."""
        indices = set()
        for perm in permutations(range(4)):
            idx = permutation_to_index(list(perm), 4)
            assert 0 <= idx < 24, f"Index {idx} out of range for perm {perm}"
            indices.add(idx)
        assert len(indices) == 24, f"Expected 24 unique indices, got {len(indices)}"


# ===========================================================================
# TestExtractOrdinalPatterns
# ===========================================================================
class TestExtractOrdinalPatterns:
    def test_monotone_one_pattern(self) -> None:
        """Monotone sequence [0..9] with D=3 should yield 8 identical patterns, no ties."""
        patterns, tied_count, tie_rate = extract_ordinal_patterns(
            list(range(10)), D=3, tau=1
        )
        assert len(patterns) == 8
        assert len(set(patterns)) == 1, "All patterns should be identical for monotone"
        assert tied_count == 0
        assert tie_rate == 0.0

    def test_too_short(self) -> None:
        """Sequence shorter than D should produce empty patterns."""
        patterns, tied_count, tie_rate = extract_ordinal_patterns(
            [0.1, 0.2], D=3, tau=1
        )
        assert patterns == []
        assert tied_count == 0
        assert tie_rate == 0.0

    def test_ties_excluded(self) -> None:
        """Constant sequence should have all windows tied."""
        patterns, tied_count, tie_rate = extract_ordinal_patterns(
            [0.5] * 10, D=3, tau=1
        )
        assert patterns == []
        assert tied_count == 8
        assert tie_rate == 1.0

    def test_delay_tau(self) -> None:
        """With tau=2, windows should skip elements."""
        # Sequence: [10, 20, 30, 40, 50, 60, 70]
        # D=3, tau=2 -> windows use indices [i, i+2, i+4]
        # n_patterns = 7 - (3-1)*2 = 3
        # Window 0: [10, 30, 50] -> monotone
        # Window 1: [20, 40, 60] -> monotone
        # Window 2: [30, 50, 70] -> monotone
        patterns, tied_count, tie_rate = extract_ordinal_patterns(
            [10, 20, 30, 40, 50, 60, 70], D=3, tau=2
        )
        assert len(patterns) == 3
        assert tied_count == 0
        assert len(set(patterns)) == 1, "All should be the same monotone pattern"

    def test_pattern_indices_valid_range(self) -> None:
        """All pattern indices should be in [0, D!)."""
        seq = _CASES["random_uniform"]["sequence"]
        patterns, _, _ = extract_ordinal_patterns(seq, D=3, tau=1)
        d_factorial = math.factorial(3)
        for idx in patterns:
            assert 0 <= idx < d_factorial, f"Pattern index {idx} out of range"


# ===========================================================================
# TestPermutationEntropy
# ===========================================================================
class TestPermutationEntropy:
    def test_monotone_low_pe(self) -> None:
        """Monotone sequence should have very low PE (one dominant pattern)."""
        pe, tie_rate, _ = permutation_entropy(list(range(20)), D=3, tau=1)
        assert pe is not None
        assert pe < 0.1, f"Expected PE < 0.1 for monotone, got {pe}"

    def test_diverse_high_pe(self) -> None:
        """Random-looking fixture should have high PE (diverse patterns)."""
        seq = _CASES["random_uniform"]["sequence"]
        pe, tie_rate, _ = permutation_entropy(seq, D=3, tau=1)
        assert pe is not None
        assert pe > 0.8, f"Expected PE > 0.8 for diverse sequence, got {pe}"

    def test_pe_range(self) -> None:
        """PE should always be in [0, 1] for any valid sequence."""
        seq = _CASES["random_uniform"]["sequence"]
        pe, _, _ = permutation_entropy(seq, D=3, tau=1)
        assert pe is not None
        assert 0.0 <= pe <= 1.0, f"PE out of range: {pe}"

    def test_flat_ties_excluded(self) -> None:
        """Near-flat sequence with epsilon=1e-4 should have high tie_rate."""
        seq = _CASES["flat_with_noise"]["sequence"]
        pe, tie_rate, _ = permutation_entropy(seq, D=3, tau=1, epsilon=1e-4)
        assert tie_rate > 0.5, f"Expected high tie_rate, got {tie_rate}"

    def test_too_short_returns_none(self) -> None:
        """Sequence shorter than D should return PE=None."""
        pe, tie_rate, pattern_counts = permutation_entropy([0.5, 0.6], D=3, tau=1)
        assert pe is None
        assert pattern_counts == {}

    def test_all_ties_returns_none(self) -> None:
        """All-constant sequence should return PE=None with tie_rate=1.0."""
        pe, tie_rate, pattern_counts = permutation_entropy([0.5] * 20, D=3, tau=1)
        assert pe is None
        assert tie_rate == 1.0
        assert pattern_counts == {}

    def test_pattern_counts_sum(self) -> None:
        """Sum of pattern_counts values should equal number of strict-order patterns."""
        seq = _CASES["random_uniform"]["sequence"]
        pe, tie_rate, pattern_counts = permutation_entropy(seq, D=3, tau=1)
        patterns, _, _ = extract_ordinal_patterns(seq, D=3, tau=1)
        assert sum(pattern_counts.values()) == len(patterns)
