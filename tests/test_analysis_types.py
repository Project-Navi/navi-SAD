"""Tests for analysis result types."""

from __future__ import annotations

import json

import pytest

from navi_sad.analysis.types import (
    EligibilityCell,
    EligibilityTable,
    PermutationNullConfig,
    PermutationNullResult,
    RecurrenceNullReport,
    RecurrenceProfile,
    RecurrenceStatistic,
)


class TestEligibilityCell:
    def test_construction(self) -> None:
        cell = EligibilityCell(
            mode="raw",
            segment="full",
            n_correct_eligible=28,
            n_incorrect_eligible=9,
            n_correct_pe_present=28,
            n_incorrect_pe_present=9,
            n_correct_total=28,
            n_incorrect_total=9,
        )
        assert cell.mode == "raw"
        assert cell.n_correct_eligible == 28

    def test_frozen(self) -> None:
        cell = EligibilityCell(
            mode="raw",
            segment="full",
            n_correct_eligible=28,
            n_incorrect_eligible=9,
            n_correct_pe_present=28,
            n_incorrect_pe_present=9,
            n_correct_total=28,
            n_incorrect_total=9,
        )
        with pytest.raises(AttributeError):
            cell.mode = "diff"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        cell = EligibilityCell(
            mode="raw",
            segment="full",
            n_correct_eligible=28,
            n_incorrect_eligible=9,
            n_correct_pe_present=28,
            n_incorrect_pe_present=9,
            n_correct_total=28,
            n_incorrect_total=9,
        )
        d = cell.to_dict()
        assert d["mode"] == "raw"
        assert d["n_correct_eligible"] == 28


class TestPermutationNullConfig:
    def test_defaults(self) -> None:
        cfg = PermutationNullConfig()
        assert cfg.n_permutations == 10_000
        assert cfg.d_threshold == 0.5
        assert cfg.min_combos == 3
        assert cfg.n_bins == 2
        assert cfg.seed == 42

    def test_custom(self) -> None:
        cfg = PermutationNullConfig(n_permutations=5000, seed=99)
        assert cfg.n_permutations == 5000
        assert cfg.seed == 99

    def test_zero_permutations_raises(self) -> None:
        with pytest.raises(ValueError, match="n_permutations"):
            PermutationNullConfig(n_permutations=0)

    def test_negative_d_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="d_threshold"):
            PermutationNullConfig(d_threshold=-0.5)

    def test_zero_min_combos_raises(self) -> None:
        with pytest.raises(ValueError, match="min_combos"):
            PermutationNullConfig(min_combos=0)

    def test_zero_n_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            PermutationNullConfig(n_bins=0)


class TestRecurrenceStatistic:
    def test_construction(self) -> None:
        stat = RecurrenceStatistic(
            d_threshold=0.5,
            min_combos=3,
            recurring_head_count=338,
            total_heads=1024,
            per_head_combo_counts={(0, 0): 5, (0, 1): 2},
        )
        assert stat.recurring_head_count == 338

    def test_to_dict_tuple_keys_become_strings(self) -> None:
        stat = RecurrenceStatistic(
            d_threshold=0.5,
            min_combos=3,
            recurring_head_count=1,
            total_heads=2,
            per_head_combo_counts={(0, 0): 5},
        )
        d = stat.to_dict()
        assert "0,0" in d["per_head_combo_counts"]


class TestRecurrenceProfile:
    def test_at_threshold(self) -> None:
        profile = RecurrenceProfile(counts_at_level={1: 800, 3: 338, 7: 11, 12: 0})
        assert profile.counts_at_level[3] == 338


class TestPermutationNullResult:
    def test_construction(self) -> None:
        result = PermutationNullResult(
            observed=338,
            null_counts=[300, 310, 320],
            p_value=0.25,
            expected_under_null=310.0,
            null_mean=310.0,
            null_std=10.0,
            null_min=300,
            null_max=320,
            null_percentiles={5: 300, 50: 310, 95: 320},
        )
        assert result.p_value == 0.25


class TestRecurrenceNullReport:
    def test_to_dict_round_trip(self) -> None:
        """Verify to_dict produces JSON-serializable output."""
        report = RecurrenceNullReport(
            config=PermutationNullConfig(n_permutations=10),
            eligibility=EligibilityTable(
                cells=[EligibilityCell("raw", "full", 5, 3, 5, 3, 5, 3)],
                n_correct=5,
                n_incorrect=3,
            ),
            observed=RecurrenceStatistic(
                0.5, 3, 2, 4, {(0, 0): 3, (0, 1): 1, (1, 0): 4, (1, 1): 0}
            ),
            observed_profile=RecurrenceProfile({1: 3, 3: 2, 7: 0}),
            null_at_min_combos=PermutationNullResult(
                observed=2,
                null_counts=[1, 2, 3],
                p_value=0.5,
                expected_under_null=2.0,
                null_mean=2.0,
                null_std=1.0,
                null_min=1,
                null_max=3,
                null_percentiles={50: 2},
            ),
            null_at_seven=PermutationNullResult(
                observed=0,
                null_counts=[0, 0, 0],
                p_value=1.0,
                expected_under_null=0.0,
                null_mean=0.0,
                null_std=0.0,
                null_min=0,
                null_max=0,
                null_percentiles={50: 0},
            ),
            bin_boundaries=[126],
            bin_counts={
                "0": {"correct": 3, "incorrect": 2},
                "1": {"correct": 2, "incorrect": 1},
            },
        )
        serialized = json.dumps(report.to_dict())
        assert isinstance(serialized, str)

    def test_none_eligibility(self) -> None:
        """Report works with eligibility=None."""
        report = RecurrenceNullReport(
            config=PermutationNullConfig(n_permutations=10),
            eligibility=None,
            observed=RecurrenceStatistic(0.5, 3, 0, 1, {(0, 0): 0}),
            observed_profile=RecurrenceProfile({1: 0}),
            null_at_min_combos=PermutationNullResult(
                observed=0,
                null_counts=[0],
                p_value=1.0,
                expected_under_null=0.0,
                null_mean=0.0,
                null_std=0.0,
                null_min=0,
                null_max=0,
                null_percentiles={50: 0},
            ),
            null_at_seven=PermutationNullResult(
                observed=0,
                null_counts=[0],
                p_value=1.0,
                expected_under_null=0.0,
                null_mean=0.0,
                null_std=0.0,
                null_min=0,
                null_max=0,
                null_percentiles={50: 0},
            ),
            bin_boundaries=[],
            bin_counts={},
        )
        d = report.to_dict()
        assert d["eligibility"] is None
