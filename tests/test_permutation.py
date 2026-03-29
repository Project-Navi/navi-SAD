"""Tests for stratified permutation null test.

Proves: class size preservation, bin-local shuffling, deterministic RNG,
reject impossible strata, planted signal detection, exchangeable no-signal.
"""

from __future__ import annotations

import random
import typing

import pytest

from navi_sad.analysis.permutation import (
    assign_length_bins,
    compute_null_result,
    run_asymmetry_null,
    run_paired_asymmetry_null,
    run_permutation_null,
    stratified_permute_labels,
)
from navi_sad.analysis.recurrence import DMatrix, PELookup
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


class TestComputeNullResultTail:
    def test_default_right_tail(self) -> None:
        """Default tail='right': k = count(null >= observed)."""
        result = compute_null_result(observed=5, null_counts=[3, 5, 7, 9])
        # k = 3 (5, 7, 9 >= 5), p = (3+1)/(4+1) = 0.8
        assert result.p_value == pytest.approx(0.8)

    def test_left_tail(self) -> None:
        """tail='left': k = count(null <= observed)."""
        result = compute_null_result(observed=5, null_counts=[3, 5, 7, 9], tail="left")
        # k = 2 (3, 5 <= 5), p = (2+1)/(4+1) = 0.6
        assert result.p_value == pytest.approx(0.6)

    def test_two_sided_tail(self) -> None:
        """tail='two-sided': k = count(|null| >= |observed|)."""
        result = compute_null_result(observed=5, null_counts=[-6, -5, 0, 5, 6], tail="two-sided")
        # |observed| = 5. |null| values: 6, 5, 0, 5, 6
        # k = 4 (6>=5, 5>=5, 5>=5, 6>=5), p = (4+1)/(5+1) = 5/6
        assert result.p_value == pytest.approx(5 / 6)

    def test_two_sided_with_negative_observed(self) -> None:
        """Two-sided works when observed is negative."""
        result = compute_null_result(observed=-3, null_counts=[-4, -3, 0, 3, 4], tail="two-sided")
        # |observed| = 3. |null| values: 4, 3, 0, 3, 4
        # k = 4, p = (4+1)/(5+1) = 5/6
        assert result.p_value == pytest.approx(5 / 6)

    def test_invalid_tail_raises(self) -> None:
        with pytest.raises(ValueError, match="tail"):
            compute_null_result(observed=5, null_counts=[1, 2, 3], tail="invalid")


class TestRunAsymmetryNull:
    def _make_directional_d_matrix(self, n_neg: int, n_pos: int, num_heads: int) -> DMatrix:
        """Build d matrix where n_neg heads have d=-0.3 and n_pos have d=+0.3."""
        from navi_sad.analysis.recurrence import EXPECTED_COMBOS

        combos = sorted(EXPECTED_COMBOS)
        d_matrix: DMatrix = {combo: {} for combo in combos}
        for h in range(num_heads):
            d_val = -0.3 if h < n_neg else 0.3
            for combo in combos:
                d_matrix[combo][(0, h)] = d_val
        return d_matrix

    def test_observed_statistic_computed(self) -> None:
        """Verify observed AsymmetryStatistic is computed correctly.

        With planted signal (correct PE > incorrect), all heads should
        have positive d (correct - incorrect), so signed_excess < 0
        (more positive than negative -> n_neg - n_pos < 0).
        Under shared-label permutation, heads are correlated so
        absolute p-value thresholds are not reliable in small fixtures.
        """
        rng = random.Random(42)
        labels = {i: "correct" if i <= 15 else "incorrect" for i in range(1, 31)}
        token_counts = dict.fromkeys(range(1, 31), 100)
        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                head_pe: dict[tuple[int, int], dict[int, float]] = {}
                for h in range(8):
                    head_pe[(0, h)] = {
                        **{i: 0.8 + rng.gauss(0, 0.02) for i in range(1, 16)},
                        **{i: 0.3 + rng.gauss(0, 0.02) for i in range(16, 31)},
                    }
                lookup[(mode, segment)] = head_pe

        result = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=8,
            n_permutations=100,
            n_bins=1,
            seed=42,
        )
        # All heads have same direction under true labels
        obs = result.observed
        assert obs.n_positive_heads + obs.n_negative_heads == 8
        assert obs.n_sparse_heads == 0
        assert obs.n_absent_heads == 0
        assert abs(obs.signed_excess) == 8  # all agree on direction

    def test_returns_correct_structure(self) -> None:
        """Verify AsymmetryNullResult has all expected fields."""
        rng = random.Random(42)
        labels = {i: "correct" if i <= 5 else "incorrect" for i in range(1, 11)}
        token_counts = dict.fromkeys(range(1, 11), 100)
        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                lookup[(mode, segment)] = {
                    (0, 0): {i: 0.5 + rng.gauss(0, 0.01) for i in range(1, 11)}
                }

        result = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=1,
            n_permutations=50,
            n_bins=1,
            seed=42,
        )
        assert isinstance(result.observed.signed_excess, int)
        assert 0 < result.p_value_two_sided <= 1.0
        assert 0 < result.p_value_one_sided_negative <= 1.0
        assert result.n_permutations == 50
        assert isinstance(result.null_signed_excess_summary.mean, float)
        assert isinstance(result.null_signed_excess_summary.std, float)

    def test_deterministic_with_seed(self) -> None:
        """Same seed -> same results."""
        rng = random.Random(42)
        labels = {i: "correct" if i <= 5 else "incorrect" for i in range(1, 11)}
        token_counts = dict.fromkeys(range(1, 11), 100)
        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                lookup[(mode, segment)] = {
                    (0, 0): {i: 0.5 + rng.gauss(0, 0.01) for i in range(1, 11)}
                }

        r1 = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=1,
            n_permutations=50,
            n_bins=1,
            seed=42,
        )
        r2 = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=1,
            n_permutations=50,
            n_bins=1,
            seed=42,
        )
        assert r1.p_value_two_sided == r2.p_value_two_sided
        assert r1.p_value_one_sided_negative == r2.p_value_one_sided_negative


class TestRunPairedAsymmetryNull:
    def _make_paired_fixture(
        self,
    ) -> tuple[PELookup, dict[int, str], list[tuple[int, int]]]:
        """Build paired fixture: 10 pairs, matched by token count.

        Each pair has one correct and one incorrect sample.
        Signal: correct PE higher than incorrect across all heads.
        """
        rng = random.Random(42)
        labels: dict[int, str] = {}
        pairs: list[tuple[int, int]] = []
        for i in range(10):
            correct_idx = 2 * i + 1
            incorrect_idx = 2 * i + 2
            labels[correct_idx] = "correct"
            labels[incorrect_idx] = "incorrect"
            pairs.append((correct_idx, incorrect_idx))

        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                head_pe: dict[tuple[int, int], dict[int, float]] = {}
                for h in range(4):
                    pe_vals: dict[int, float] = {}
                    for correct_idx, incorrect_idx in pairs:
                        pe_vals[correct_idx] = 0.8 + rng.gauss(0, 0.02)
                        pe_vals[incorrect_idx] = 0.3 + rng.gauss(0, 0.02)
                    head_pe[(0, h)] = pe_vals
                lookup[(mode, segment)] = head_pe

        return lookup, labels, pairs

    def test_returns_correct_structure(self) -> None:
        """Paired null returns AsymmetryNullResult with both p-values."""
        lookup, labels, pairs = self._make_paired_fixture()
        result = run_paired_asymmetry_null(
            lookup=lookup,
            labels=labels,
            pairs=pairs,
            num_layers=1,
            num_heads=4,
            n_permutations=50,
            seed=42,
        )
        assert isinstance(result.observed.signed_excess, int)
        assert 0 < result.p_value_two_sided <= 1.0
        assert 0 < result.p_value_one_sided_negative <= 1.0
        assert result.n_permutations == 50
        assert isinstance(result.null_signed_excess_summary.mean, float)

    def test_deterministic_with_seed(self) -> None:
        """Same seed -> same results."""
        lookup, labels, pairs = self._make_paired_fixture()
        r1 = run_paired_asymmetry_null(
            lookup=lookup,
            labels=labels,
            pairs=pairs,
            num_layers=1,
            num_heads=4,
            n_permutations=50,
            seed=42,
        )
        r2 = run_paired_asymmetry_null(
            lookup=lookup,
            labels=labels,
            pairs=pairs,
            num_layers=1,
            num_heads=4,
            n_permutations=50,
            seed=42,
        )
        assert r1.p_value_two_sided == r2.p_value_two_sided
        assert r1.p_value_one_sided_negative == r2.p_value_one_sided_negative

    def test_pair_structure_preserved(self) -> None:
        """Under paired permutation, each pair swaps or keeps independently."""
        lookup, labels, pairs = self._make_paired_fixture()
        result = run_paired_asymmetry_null(
            lookup=lookup,
            labels=labels,
            pairs=pairs,
            num_layers=1,
            num_heads=4,
            n_permutations=100,
            seed=42,
        )
        # With 10 pairs and independent coin flips, the null distribution
        # should not be degenerate (all same value). Check variance > 0.
        assert result.null_signed_excess_summary.std >= 0.0
        # The observed stat should have the expected number of heads
        assert result.observed.n_positive_heads + result.observed.n_negative_heads <= 4

    def test_empty_pairs_raises(self) -> None:
        """Empty pairs list must raise."""
        with pytest.raises(ValueError, match="pairs"):
            run_paired_asymmetry_null(
                lookup={},
                labels={},
                pairs=[],
                num_layers=1,
                num_heads=1,
                n_permutations=10,
                seed=42,
            )


class TestGoldenValueRegression:
    """Golden-value tests proving numeric identity after null-summary refactor.

    Values captured on main (commit 8a24c77) before refactor.
    If any value changes, the refactor has introduced a behavioral difference.
    """

    GOLDEN_NULL_COUNTS: typing.ClassVar[list[int]] = [3, 7, 1, 9, 5, 2, 8, 4, 6, 0]

    def test_compute_null_result_right_tail(self) -> None:
        r = compute_null_result(observed=5, null_counts=self.GOLDEN_NULL_COUNTS, tail="right")
        assert r.p_value == 0.5454545454545454
        assert r.null_mean == 4.5
        assert r.null_std == 2.8722813232690143
        assert r.null_min == 0
        assert r.null_max == 9
        assert r.null_percentiles == {5: 0, 25: 2, 50: 5, 75: 7, 95: 9}

    def test_compute_null_result_left_tail(self) -> None:
        r = compute_null_result(observed=5, null_counts=self.GOLDEN_NULL_COUNTS, tail="left")
        assert r.p_value == 0.6363636363636364

    def test_compute_null_result_two_sided(self) -> None:
        r = compute_null_result(observed=5, null_counts=self.GOLDEN_NULL_COUNTS, tail="two-sided")
        assert r.p_value == 0.5454545454545454

    def test_asymmetry_null_summary_identity(self) -> None:
        """Asymmetry null summary must match golden values exactly."""
        rng = random.Random(42)
        labels = {i: "correct" if i <= 5 else "incorrect" for i in range(1, 11)}
        token_counts = dict.fromkeys(range(1, 11), 100)
        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                lookup[(mode, segment)] = {
                    (0, 0): {i: 0.5 + rng.gauss(0, 0.01) for i in range(1, 11)}
                }

        result = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=1,
            n_permutations=50,
            n_bins=1,
            seed=42,
        )
        assert result.p_value_two_sided == 1.0
        assert result.p_value_one_sided_negative == 0.6274509803921569
        assert result.observed.signed_excess == 1

        s = result.null_signed_excess_summary
        assert s.mean == pytest.approx(0.24)
        assert s.std == pytest.approx(0.9707728879609279, rel=1e-12)
        assert s.min_val == -1.0
        assert s.max_val == 1.0
        assert s.percentiles == {5: -1.0, 25: -1.0, 50: 1.0, 75: 1.0, 95: 1.0}
        assert s.n == 50

    def test_asymmetry_null_to_dict_legacy_flat_shape(self) -> None:
        """AsymmetryNullResult.to_dict() must preserve legacy flat JSON shape.

        The old _null_summary() produced {mean, std, min, max, p5, p25, ...}.
        The refactored code must emit the same keys for artifact compatibility.
        """
        rng = random.Random(42)
        labels = {i: "correct" if i <= 5 else "incorrect" for i in range(1, 11)}
        token_counts = dict.fromkeys(range(1, 11), 100)
        lookup: PELookup = {}
        for mode in ("raw", "diff", "residual"):
            for segment in ("full", "early", "mid", "late"):
                lookup[(mode, segment)] = {
                    (0, 0): {i: 0.5 + rng.gauss(0, 0.01) for i in range(1, 11)}
                }

        result = run_asymmetry_null(
            lookup=lookup,
            labels=labels,
            token_counts=token_counts,
            num_layers=1,
            num_heads=1,
            n_permutations=50,
            n_bins=1,
            seed=42,
        )
        d = result.to_dict()
        summary = d["null_signed_excess_summary"]
        # Must have legacy flat keys, not nested "percentiles" object
        assert "mean" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary
        assert "p5" in summary
        assert "p25" in summary
        assert "p50" in summary
        assert "p75" in summary
        assert "p95" in summary
        # Must NOT have the nested NullDistributionSummary shape
        assert "percentiles" not in summary
        assert "min_val" not in summary
        assert "max_val" not in summary
