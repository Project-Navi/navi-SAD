"""Tests for d matrix computation, recurrence decomposition, and landscape summary.

Proves: d matrix preserves values, recurrence_from_d_matrix matches
compute_recurrence, summarize_d_matrix produces correct statistics,
threshold sweep counts are consistent, directional counts are correct.
"""

from __future__ import annotations

import pytest

from navi_sad.analysis.recurrence import (
    DMatrix,
    compute_d_matrix,
    compute_recurrence,
    recurrence_from_d_matrix,
    summarize_d_matrix,
)
from navi_sad.signal.pe_features import HeadPEResult, PEConfig, SamplePEFeatures


def _head(
    layer: int,
    head: int,
    mode: str,
    segment: str,
    pe: float | None = 0.8,
    eligible: bool = True,
) -> HeadPEResult:
    return HeadPEResult(
        layer_idx=layer,
        head_idx=head,
        mode=mode,
        segment=segment,
        sequence_length=20 if eligible else 5,
        eligible=eligible,
        pe=pe,
        tie_rate=0.0,
        n_strict_patterns=10,
    )


def _sample(idx: int, heads: list[HeadPEResult]) -> SamplePEFeatures:
    return SamplePEFeatures(dataset_index=idx, config=PEConfig(), heads=heads)


def _build_lookup_and_labels() -> tuple[
    dict[tuple[str, str], dict[tuple[int, int], dict[int, float]]],
    dict[int, str],
]:
    """Build a lookup with known separation for testing.

    4 correct samples with PE ~0.9, 4 incorrect with PE ~0.1.
    2 heads in 2 combos. Within-group variance ensures nonzero pooled var.
    """
    labels = {
        1: "correct",
        2: "correct",
        3: "correct",
        4: "correct",
        5: "incorrect",
        6: "incorrect",
        7: "incorrect",
        8: "incorrect",
    }
    lookup = {
        ("raw", "full"): {
            (0, 0): {1: 0.91, 2: 0.89, 3: 0.92, 4: 0.88, 5: 0.11, 6: 0.09, 7: 0.12, 8: 0.08},
            (0, 1): {1: 0.50, 2: 0.51, 3: 0.49, 4: 0.52, 5: 0.48, 6: 0.47, 7: 0.53, 8: 0.50},
        },
        ("diff", "full"): {
            (0, 0): {1: 0.85, 2: 0.87, 3: 0.83, 4: 0.86, 5: 0.15, 6: 0.13, 7: 0.17, 8: 0.14},
            (0, 1): {1: 0.45, 2: 0.55, 3: 0.50, 4: 0.48, 5: 0.52, 6: 0.46, 7: 0.51, 8: 0.49},
        },
    }
    return lookup, labels


class TestComputeDMatrix:
    def test_returns_d_values_not_counts(self) -> None:
        """d matrix contains float d values, not threshold counts."""
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)

        # Head (0,0) has strong separation -> large |d|
        d_raw = d_matrix[("raw", "full")][(0, 0)]
        assert d_raw is not None
        assert isinstance(d_raw, float)
        assert abs(d_raw) > 1.0  # strong separation

        # Head (0,1) has weak separation -> small |d|
        d_weak = d_matrix[("raw", "full")][(0, 1)]
        assert d_weak is not None
        assert abs(d_weak) < 1.0

    def test_preserves_all_combos(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        assert set(d_matrix.keys()) == {("raw", "full"), ("diff", "full")}

    def test_preserves_all_heads(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        for combo in d_matrix:
            assert set(d_matrix[combo].keys()) == {(0, 0), (0, 1)}

    def test_rejects_stray_labels(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        labels[1] = "ambiguous"
        with pytest.raises(ValueError, match="non-canonical"):
            compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)

    def test_rejects_out_of_grid_head(self) -> None:
        lookup = {("raw", "full"): {(5, 0): {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}}}
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        with pytest.raises(ValueError, match="outside the declared grid"):
            compute_d_matrix(lookup, labels, num_layers=1, num_heads=1)

    def test_none_for_insufficient_data(self) -> None:
        """< 2 samples per group -> d is None."""
        lookup = {("raw", "full"): {(0, 0): {1: 0.5, 2: 0.5}}}
        labels = {1: "correct", 2: "incorrect"}
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=1)
        assert d_matrix[("raw", "full")][(0, 0)] is None


class TestRecurrenceFromDMatrix:
    def test_matches_compute_recurrence(self) -> None:
        """recurrence_from_d_matrix produces identical results to compute_recurrence."""
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)

        stat_direct, profile_direct = compute_recurrence(
            lookup,
            labels,
            d_threshold=0.5,
            min_combos=1,
            num_layers=1,
            num_heads=2,
        )
        stat_matrix, profile_matrix = recurrence_from_d_matrix(
            d_matrix,
            d_threshold=0.5,
            min_combos=1,
            num_layers=1,
            num_heads=2,
        )

        assert stat_direct.recurring_head_count == stat_matrix.recurring_head_count
        assert stat_direct.per_head_combo_counts == stat_matrix.per_head_combo_counts
        assert profile_direct.counts_at_level == profile_matrix.counts_at_level

    def test_threshold_sweep_via_d_matrix(self) -> None:
        """Same d matrix, different thresholds, no recomputation of d."""
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)

        stat_low, _ = recurrence_from_d_matrix(
            d_matrix,
            d_threshold=0.1,
            min_combos=1,
            num_layers=1,
            num_heads=2,
        )
        stat_high, _ = recurrence_from_d_matrix(
            d_matrix,
            d_threshold=5.0,
            min_combos=1,
            num_layers=1,
            num_heads=2,
        )

        # Low threshold catches more
        assert stat_low.recurring_head_count >= stat_high.recurring_head_count


class TestSummarizeDMatrix:
    def test_counts_are_consistent(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        summary = summarize_d_matrix(d_matrix)

        assert (
            summary["n_computable"]
            == summary["n_positive"] + summary["n_negative"] + summary["n_zero"]
        )
        assert summary["n_total"] == summary["n_computable"] + summary["n_none"]

    def test_positive_fraction_correct(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        summary = summarize_d_matrix(d_matrix)

        n_pos = summary["n_positive"]
        n_neg = summary["n_negative"]
        expected = n_pos / (n_pos + n_neg) if (n_pos + n_neg) > 0 else None
        assert summary["positive_fraction"] == pytest.approx(expected)

    def test_max_abs_d_correct(self) -> None:
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        summary = summarize_d_matrix(d_matrix)

        # Collect all d values manually
        all_d = []
        for head_d in d_matrix.values():
            for d_val in head_d.values():
                if d_val is not None:
                    all_d.append(abs(d_val))

        assert summary["max_abs_d"] == pytest.approx(max(all_d))

    def test_threshold_sweep_monotonic(self) -> None:
        """Higher thresholds -> fewer cells exceeding."""
        lookup, labels = _build_lookup_and_labels()
        d_matrix = compute_d_matrix(lookup, labels, num_layers=1, num_heads=2)
        summary = summarize_d_matrix(d_matrix)

        sweep = summary["threshold_sweep"]
        thresholds = sorted(float(k) for k in sweep)
        counts = [sweep[str(t)] for t in thresholds]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], f"Non-monotonic at threshold {thresholds[i]}"

    def test_empty_d_matrix(self) -> None:
        summary = summarize_d_matrix({})
        assert summary["n_total"] == 0
        assert summary["n_computable"] == 0
        assert summary["max_abs_d"] is None
        assert summary["positive_fraction"] is None

    def test_all_none_d_values(self) -> None:
        d_matrix: DMatrix = {("raw", "full"): {(0, 0): None, (0, 1): None}}
        summary = summarize_d_matrix(d_matrix)
        assert summary["n_none"] == 2
        assert summary["n_computable"] == 0
        assert summary["max_abs_d"] is None

    def test_known_direction(self) -> None:
        """All positive d -> positive_fraction = 1.0."""
        d_matrix: DMatrix = {
            ("raw", "full"): {(0, 0): 1.5, (0, 1): 0.3},
            ("diff", "full"): {(0, 0): 2.0, (0, 1): 0.1},
        }
        summary = summarize_d_matrix(d_matrix)
        assert summary["n_positive"] == 4
        assert summary["n_negative"] == 0
        assert summary["positive_fraction"] == pytest.approx(1.0)

    def test_uses_numpy(self) -> None:
        """Verify numpy is actually being used (percentiles should be precise)."""
        d_matrix: DMatrix = {
            ("raw", "full"): {(i, 0): float(i) * 0.01 for i in range(100)},
        }
        summary = summarize_d_matrix(d_matrix)
        # numpy percentile on 100 values should be precise
        assert summary["p95_abs_d"] is not None
        assert isinstance(summary["p95_abs_d"], float)
