"""Tests for recurrence statistic computation.

Proves: correct counting on hand-built fixtures, threshold boundary,
absent/ineligible cell handling, no RNG involved.
"""

from __future__ import annotations

import pytest

from navi_sad.analysis.recurrence import (
    build_pe_lookup,
    compute_combo_cohens_d,
    compute_recurrence,
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


class TestBuildPELookup:
    def test_basic(self) -> None:
        samples = {
            1: _sample(1, [_head(0, 0, "raw", "full", pe=0.8)]),
            2: _sample(2, [_head(0, 0, "raw", "full", pe=0.6)]),
        }
        lookup = build_pe_lookup(samples)
        assert lookup[("raw", "full")][(0, 0)] == {1: 0.8, 2: 0.6}

    def test_ineligible_excluded(self) -> None:
        samples = {
            1: _sample(1, [_head(0, 0, "raw", "full", eligible=False, pe=None)]),
        }
        lookup = build_pe_lookup(samples)
        # Ineligible head not in lookup
        assert (0, 0) not in lookup.get(("raw", "full"), {})

    def test_pe_none_excluded(self) -> None:
        """Eligible but pe=None (all ties) — excluded from lookup."""
        samples = {
            1: _sample(1, [_head(0, 0, "raw", "full", eligible=True, pe=None)]),
        }
        lookup = build_pe_lookup(samples)
        assert (0, 0) not in lookup.get(("raw", "full"), {})


class TestComboCohensd:
    def test_known_d_value(self) -> None:
        """Two groups with known separation.

        A=[1.0, 2.0], B=[0.0, 0.0]
        mean_a=1.5, var_a=0.5, mean_b=0.0, var_b=0.0
        pooled = (0.5 + 0.0) / 2 = 0.25, d = 1.5 / 0.5 = 3.0
        """
        import pytest

        head_pe: dict[tuple[int, int], dict[int, float]] = {
            (0, 0): {1: 1.0, 2: 2.0, 3: 0.0, 4: 0.0}
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        d_values = compute_combo_cohens_d(head_pe, labels)
        assert (0, 0) in d_values
        assert d_values[(0, 0)] == pytest.approx(3.0)

    def test_insufficient_group_returns_none(self) -> None:
        """< 2 samples in a group -> None."""
        head_pe: dict[tuple[int, int], dict[int, float]] = {(0, 0): {1: 1.0, 2: 0.5}}
        labels = {1: "correct", 2: "incorrect"}
        d_values = compute_combo_cohens_d(head_pe, labels)
        assert d_values[(0, 0)] is None

    def test_empty_head_pe(self) -> None:
        d_values = compute_combo_cohens_d({}, {})
        assert d_values == {}

    def test_near_zero_variance_returns_none(self) -> None:
        """Near-zero pooled variance must return None, not huge d."""
        # Two groups where values differ by ~1e-15 (float rounding)
        eps = 1e-16
        head_pe: dict[tuple[int, int], dict[int, float]] = {
            (0, 0): {1: 0.5, 2: 0.5 + eps, 3: 0.5, 4: 0.5 + eps}
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        d_values = compute_combo_cohens_d(head_pe, labels)
        assert d_values[(0, 0)] is None


class TestComputeRecurrence:
    def test_hand_built_recurrence(self) -> None:
        """Head (0,0) has |d|>0.5 in 3 combos, head (0,1) in 2 combos."""
        # Need within-group variance to avoid pooled_var=0 (which yields d=None).
        lookup = {
            ("raw", "full"): {
                (0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
                (0, 1): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
            },
            ("raw", "early"): {
                (0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
                (0, 1): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
            },
            ("diff", "full"): {
                (0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
                # (0, 1) not present -> absent
            },
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        stat, _ = compute_recurrence(
            lookup,
            labels,
            d_threshold=0.5,
            min_combos=3,
            num_layers=1,
            num_heads=2,
        )
        assert stat.recurring_head_count == 1  # only (0,0) has 3+ combos
        assert stat.per_head_combo_counts[(0, 0)] == 3
        assert stat.per_head_combo_counts[(0, 1)] == 2

    def test_threshold_strict_greater_than(self) -> None:
        """Recurrence uses strict > (not >=) on |d|.

        Tests the inequality implementation, not the production threshold
        value. Uses d=3.0 at threshold=3.0 as a proxy because constructing
        exact d=0.5 with small integer fixtures is non-trivial.
        """
        lookup = {
            ("raw", "full"): {
                (0, 0): {1: 1.0, 2: 2.0, 3: 0.0, 4: 0.0},
            },
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        # d = 3.0 for head (0,0). Threshold = 3.0 -> NOT recurring (> not >=)
        stat, _ = compute_recurrence(
            lookup,
            labels,
            d_threshold=3.0,
            min_combos=1,
            num_layers=1,
            num_heads=1,
        )
        assert stat.per_head_combo_counts[(0, 0)] == 0

    def test_absent_combo_does_not_count(self) -> None:
        """Head not present in a combo -> that combo doesn't count."""
        lookup = {
            ("raw", "full"): {(0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1}},
            ("raw", "early"): {},  # (0,0) absent
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        stat, _ = compute_recurrence(
            lookup,
            labels,
            d_threshold=0.5,
            min_combos=2,
            num_layers=1,
            num_heads=1,
        )
        assert stat.per_head_combo_counts[(0, 0)] == 1
        assert stat.recurring_head_count == 0

    def test_none_d_does_not_count(self) -> None:
        """Combo with < 2 samples per group -> d=None -> does not count."""
        lookup = {
            ("raw", "full"): {(0, 0): {1: 1.0, 2: 0.0}},  # 1 per group
            ("raw", "early"): {(0, 0): {1: 1.0, 2: 0.0}},
            ("raw", "mid"): {(0, 0): {1: 1.0, 2: 0.0}},
        }
        labels = {1: "correct", 2: "incorrect"}
        stat, _ = compute_recurrence(
            lookup,
            labels,
            d_threshold=0.5,
            min_combos=3,
            num_layers=1,
            num_heads=1,
        )
        assert stat.recurring_head_count == 0  # all d=None

    def test_profile_counts(self) -> None:
        """Profile reports head counts at each combo level."""
        lookup = {
            ("raw", "full"): {
                (0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
                (0, 1): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
            },
            ("diff", "full"): {
                (0, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},
            },
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        _, profile = compute_recurrence(
            lookup,
            labels,
            d_threshold=0.5,
            min_combos=1,
            num_layers=1,
            num_heads=2,
        )
        # (0,0) has 2 combos, (0,1) has 1 combo
        # Level 1: 2 heads >= 1 combo
        # Level 2: 1 head >= 2 combos
        assert profile.counts_at_level[1] == 2
        assert profile.counts_at_level[2] == 1

    def test_out_of_grid_head_raises(self) -> None:
        """Head outside declared grid must raise, not silently drop."""
        lookup = {
            ("raw", "full"): {
                (5, 0): {1: 0.9, 2: 1.1, 3: 0.0, 4: 0.1},  # layer 5, but grid is 1x1
            },
        }
        labels = {1: "correct", 2: "correct", 3: "incorrect", 4: "incorrect"}
        with pytest.raises(ValueError, match="outside the declared grid"):
            compute_recurrence(
                lookup,
                labels,
                d_threshold=0.5,
                min_combos=1,
                num_layers=1,
                num_heads=1,
            )

    def test_stray_labels_raises(self) -> None:
        """Non-canonical labels must raise, not be silently ignored."""
        lookup: dict = {("raw", "full"): {(0, 0): {1: 0.5, 2: 0.5}}}
        labels = {1: "correct", 2: "ambiguous"}
        with pytest.raises(ValueError, match="non-canonical"):
            compute_recurrence(
                lookup,
                labels,
                d_threshold=0.5,
                min_combos=1,
                num_layers=1,
                num_heads=1,
            )

    def test_zero_dimensions_raises(self) -> None:
        """Non-positive num_layers or num_heads must raise."""
        with pytest.raises(ValueError, match="num_layers"):
            compute_recurrence(
                {},
                {},
                d_threshold=0.5,
                min_combos=1,
                num_layers=0,
                num_heads=1,
            )
