"""Typed inputs and outputs for PE recurrence null analysis.

Frozen dataclasses only. No computation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EligibilityCell:
    """Eligibility counts for one (mode, segment) combination.

    n_*_eligible: passed minimum sequence length threshold.
    n_*_pe_present: eligible AND pe is not None (contributed to Cohen's d).
    n_*_total: total samples in that class.
    """

    mode: str
    segment: str
    n_correct_eligible: int
    n_incorrect_eligible: int
    n_correct_pe_present: int
    n_incorrect_pe_present: int
    n_correct_total: int
    n_incorrect_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "segment": self.segment,
            "n_correct_eligible": self.n_correct_eligible,
            "n_incorrect_eligible": self.n_incorrect_eligible,
            "n_correct_pe_present": self.n_correct_pe_present,
            "n_incorrect_pe_present": self.n_incorrect_pe_present,
            "n_correct_total": self.n_correct_total,
            "n_incorrect_total": self.n_incorrect_total,
        }


@dataclass(frozen=True)
class EligibilityTable:
    """Full eligibility summary across all combos."""

    cells: list[EligibilityCell]
    n_correct: int
    n_incorrect: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }


@dataclass(frozen=True)
class PermutationNullConfig:
    """Configuration for the stratified permutation null test."""

    n_permutations: int = 10_000
    d_threshold: float = 0.5
    min_combos: int = 3
    n_bins: int = 2
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_permutations": self.n_permutations,
            "d_threshold": self.d_threshold,
            "min_combos": self.min_combos,
            "n_bins": self.n_bins,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RecurrenceStatistic:
    """Observed recurrence: how many heads exceed the threshold."""

    d_threshold: float
    min_combos: int
    recurring_head_count: int
    total_heads: int
    per_head_combo_counts: dict[tuple[int, int], int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_threshold": self.d_threshold,
            "min_combos": self.min_combos,
            "recurring_head_count": self.recurring_head_count,
            "total_heads": self.total_heads,
            "per_head_combo_counts": {
                f"{layer},{head}": v for (layer, head), v in self.per_head_combo_counts.items()
            },
        }


@dataclass(frozen=True)
class RecurrenceProfile:
    """Count of heads at each combo level (1 through max combos)."""

    counts_at_level: dict[int, int]

    def to_dict(self) -> dict[str, int]:
        return {str(k): v for k, v in sorted(self.counts_at_level.items())}


@dataclass(frozen=True)
class PermutationNullResult:
    """Result of a permutation null test at one threshold level."""

    observed: int
    null_counts: list[int]
    p_value: float
    expected_under_null: float
    null_mean: float
    null_std: float
    null_min: int
    null_max: int
    null_percentiles: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "expected_under_null": self.expected_under_null,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "null_min": self.null_min,
            "null_max": self.null_max,
            "null_percentiles": {str(k): v for k, v in self.null_percentiles.items()},
            "n_permutations": len(self.null_counts),
        }


@dataclass(frozen=True)
class RecurrenceNullReport:
    """Top-level report combining all analysis outputs.

    Note: frozen=True prevents attribute reassignment but list/dict fields
    remain mutable in-place. This is a Python limitation. Consumers should
    treat all fields as immutable by convention.
    """

    config: PermutationNullConfig
    eligibility: EligibilityTable | None
    observed: RecurrenceStatistic
    observed_profile: RecurrenceProfile
    null_at_min_combos: PermutationNullResult
    null_at_seven: PermutationNullResult
    bin_boundaries: list[int]
    bin_counts: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "eligibility": self.eligibility.to_dict() if self.eligibility is not None else None,
            "observed": self.observed.to_dict(),
            "observed_profile": self.observed_profile.to_dict(),
            "null_at_min_combos": self.null_at_min_combos.to_dict(),
            "null_at_seven": self.null_at_seven.to_dict(),
            "bin_boundaries": self.bin_boundaries,
            "bin_counts": self.bin_counts,
        }
