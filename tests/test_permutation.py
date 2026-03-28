"""Tests for stratified permutation null test.

Proves: class size preservation, bin-local shuffling, deterministic RNG,
reject impossible strata, planted signal detection, exchangeable no-signal.
"""

from __future__ import annotations

import random

import pytest

from navi_sad.analysis.permutation import (
    assign_length_bins,
    compute_null_result,
    run_permutation_null,
    stratified_permute_labels,
)
from navi_sad.analysis.recurrence import PELookup
from navi_sad.analysis.types import PermutationNullConfig


class TestAssignLengthBins:
    def test_median_split(self) -> None:
        token_counts = {1: 50, 2: 60, 3: 150, 4: 200}
        labels = {1: "correct", 2: "incorrect", 3: "correct", 4: "incorrect"}
        bins, boundaries = assign_length_bins(token_counts, labels, n_bins=2)
        # Upper-median rank split: boundary = sorted_counts[n//2] = 150
        assert len(boundaries) == 1
        assert boundaries[0] == 150
        assert bins[1] == bins[2]  # both below boundary
        assert bins[3] == bins[4]  # both at/above boundary
        assert bins[1] != bins[3]

    def test_single_bin(self) -> None:
        token_counts = {1: 100, 2: 200}
        labels = {1: "correct", 2: "incorrect"}
        bins, boundaries = assign_length_bins(token_counts, labels, n_bins=1)
        assert bins[1] == bins[2] == 0
        assert boundaries == []

    def test_empty_stratum_raises(self) -> None:
        """If a bin has 0 incorrect samples, raise ValueError."""
        # All incorrect in high bin, none in low bin
        token_counts = {1: 10, 2: 20, 3: 200, 4: 200}
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        # Median = 110, all incorrect above -> low bin has 0 incorrect
        with pytest.raises(ValueError, match=r"no .* samples"):
            assign_length_bins(token_counts, labels, n_bins=2)

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            assign_length_bins({}, {}, n_bins=2)


class TestStratifiedPermuteLabels:
    def test_preserves_class_sizes(self) -> None:
        labels = {
            1: "correct",
            2: "correct",
            3: "incorrect",
            4: "incorrect",
            5: "correct",
        }
        bins = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
        rng = random.Random(42)
        shuffled = stratified_permute_labels(labels, bins, rng)
        # Same total counts
        assert sum(1 for v in shuffled.values() if v == "correct") == 3
        assert sum(1 for v in shuffled.values() if v == "incorrect") == 2

    def test_preserves_bin_counts(self) -> None:
        """Class counts within each bin are preserved."""
        labels = {1: "correct", 2: "incorrect", 3: "correct", 4: "incorrect"}
        bins = {1: 0, 2: 0, 3: 1, 4: 1}
        rng = random.Random(42)
        shuffled = stratified_permute_labels(labels, bins, rng)
        # Bin 0 had 1 correct + 1 incorrect -> still has 1+1
        bin0_labels = [shuffled[i] for i in [1, 2]]
        assert sorted(bin0_labels) == ["correct", "incorrect"]
        bin1_labels = [shuffled[i] for i in [3, 4]]
        assert sorted(bin1_labels) == ["correct", "incorrect"]

    def test_deterministic_with_seed(self) -> None:
        labels = {i: "correct" if i < 5 else "incorrect" for i in range(10)}
        bins = dict.fromkeys(range(10), 0)
        r1 = stratified_permute_labels(labels, bins, random.Random(42))
        r2 = stratified_permute_labels(labels, bins, random.Random(42))
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        labels = {i: "correct" if i < 5 else "incorrect" for i in range(10)}
        bins = dict.fromkeys(range(10), 0)
        r1 = stratified_permute_labels(labels, bins, random.Random(42))
        r2 = stratified_permute_labels(labels, bins, random.Random(99))
        assert r1 != r2

    def test_insertion_order_independent(self) -> None:
        """Same items in different insertion order -> identical shuffled output."""
        labels_a = {1: "correct", 2: "incorrect", 3: "correct", 4: "incorrect"}
        labels_b = {4: "incorrect", 1: "correct", 3: "correct", 2: "incorrect"}
        bins = dict.fromkeys(range(1, 5), 0)
        r_a = stratified_permute_labels(labels_a, bins, random.Random(42))
        r_b = stratified_permute_labels(labels_b, bins, random.Random(42))
        assert r_a == r_b


class TestComputeNullResult:
    def test_empty_null_counts_raises(self) -> None:
        """Empty null_counts must raise, not produce fake p=1.0."""
        with pytest.raises(ValueError, match="empty"):
            compute_null_result(observed=5, null_counts=[])

    def test_p_value_bounds(self) -> None:
        result = compute_null_result(observed=5, null_counts=[3, 4, 5, 6, 7])
        assert 0.0 < result.p_value <= 1.0

    def test_phipson_smyth_prevents_zero(self) -> None:
        """p-value is never exactly 0 (Phipson-Smyth correction)."""
        result = compute_null_result(observed=100, null_counts=[0, 0, 0, 0, 0])
        assert result.p_value > 0.0
        # (0 + 1) / (5 + 1) = 1/6
        assert result.p_value == pytest.approx(1 / 6)

    def test_all_exceed_gives_p_near_one(self) -> None:
        result = compute_null_result(observed=0, null_counts=[10, 20, 30])
        # All null >= observed: (3+1)/(3+1) = 1.0
        assert result.p_value == pytest.approx(1.0)

    def test_expected_under_null(self) -> None:
        result = compute_null_result(observed=5, null_counts=[2, 4, 6, 8])
        assert result.expected_under_null == pytest.approx(5.0)


class TestRunPermutationNull:
    def _make_planted_signal_lookup(self) -> tuple[PELookup, dict[int, str]]:
        """Create lookup where correct and incorrect have very different PE.

        10 correct, 10 incorrect. 4 heads with independent PE distributions
        in 4 combos -> all 4 should be recurring at min_combos=3.
        Multiple heads with different PE values ensure the observed count
        (4) separates from the null (where random shuffles rarely produce
        all 4 heads exceeding d_threshold in 3+ combos).
        """
        labels: dict[int, str] = {}
        for i in range(1, 11):
            labels[i] = "correct"
        for i in range(11, 21):
            labels[i] = "incorrect"

        # Each head gets a different PE spread so they don't perfectly covary
        rng = random.Random(999)
        lookup: PELookup = {}
        for mode, segment in [
            ("raw", "full"),
            ("raw", "early"),
            ("diff", "full"),
            ("diff", "early"),
        ]:
            head_pe: dict[tuple[int, int], dict[int, float]] = {}
            for h in range(4):
                head_pe[(0, h)] = {
                    **{i: 0.85 + rng.gauss(0, 0.03) for i in range(1, 11)},
                    **{i: 0.15 + rng.gauss(0, 0.03) for i in range(11, 21)},
                }
            lookup[(mode, segment)] = head_pe

        return lookup, labels

    def _make_exchangeable_lookup(self) -> tuple[PELookup, dict[int, str]]:
        """Create lookup where correct and incorrect have same PE distribution.

        All 20 samples have PE ~ 0.5 with small noise.
        No signal -> recurrence under true labels should be typical of null.
        """
        labels: dict[int, str] = {}
        for i in range(1, 11):
            labels[i] = "correct"
        for i in range(11, 21):
            labels[i] = "incorrect"

        rng = random.Random(123)
        lookup: PELookup = {}
        for mode, segment in [
            ("raw", "full"),
            ("raw", "early"),
            ("diff", "full"),
            ("diff", "early"),
        ]:
            lookup[(mode, segment)] = {
                (0, 0): {i: 0.5 + rng.gauss(0, 0.01) for i in range(1, 21)},
            }

        return lookup, labels

    def test_planted_signal_detected(self) -> None:
        """Strong planted signal should produce low p-value.

        With n=10 per group the permutation test has limited power.
        The threshold is 0.25 (not 0.05) because this is a unit test
        proving that planted signal yields a LOWER p-value than the
        exchangeable case (which gives p > 0.5). It is not a power
        analysis.
        """
        lookup, labels = self._make_planted_signal_lookup()
        token_counts = dict.fromkeys(range(1, 21), 100)
        config = PermutationNullConfig(
            n_permutations=200,
            d_threshold=0.5,
            min_combos=3,
            n_bins=1,
            seed=42,
        )
        report = run_permutation_null(
            lookup,
            labels,
            token_counts,
            config=config,
            num_layers=1,
            num_heads=4,
        )
        assert report.null_at_min_combos.p_value < 0.25

    def test_exchangeable_no_signal(self) -> None:
        """Exchangeable data should produce high p-value."""
        lookup, labels = self._make_exchangeable_lookup()
        token_counts = dict.fromkeys(range(1, 21), 100)
        config = PermutationNullConfig(
            n_permutations=200,
            d_threshold=0.5,
            min_combos=3,
            n_bins=1,
            seed=42,
        )
        report = run_permutation_null(
            lookup,
            labels,
            token_counts,
            config=config,
            num_layers=1,
            num_heads=4,
        )
        assert report.null_at_min_combos.p_value > 0.1

    def test_deterministic_with_seed(self) -> None:
        lookup, labels = self._make_planted_signal_lookup()
        token_counts = dict.fromkeys(range(1, 21), 100)
        config = PermutationNullConfig(n_permutations=50, n_bins=1, seed=42)
        r1 = run_permutation_null(
            lookup,
            labels,
            token_counts,
            config=config,
            num_layers=1,
            num_heads=4,
        )
        r2 = run_permutation_null(
            lookup,
            labels,
            token_counts,
            config=config,
            num_layers=1,
            num_heads=4,
        )
        assert r1.null_at_min_combos.null_counts == r2.null_at_min_combos.null_counts
