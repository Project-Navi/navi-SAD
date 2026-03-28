"""Typed inputs and outputs for PE recurrence null analysis.

Frozen dataclasses only. No computation logic.
PermutationNullConfig has __post_init__ validation; the rest
are frozen containers with to_dict() serialization.
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

    def __post_init__(self) -> None:
        if self.n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {self.n_permutations}")
        if self.d_threshold <= 0:
            raise ValueError(f"d_threshold must be > 0, got {self.d_threshold}")
        if self.min_combos < 1:
            raise ValueError(f"min_combos must be >= 1, got {self.min_combos}")
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")

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
class DLandscape:
    """Full d-value landscape summary. Never discard d values.

    Computed from the d matrix over all (head, combo) cells in the
    theoretical grid. absent_cells tracks cells where PE was not
    computable (ineligible or no data), ensuring the denominator
    is the full grid, not just the present cells.
    """

    expected_total_cells: int
    present_cells: int
    absent_cells: int
    n_computable: int
    n_none: int
    n_positive: int
    n_negative: int
    n_zero: int
    positive_fraction: float | None
    max_abs_d: float | None
    mean_abs_d: float | None
    median_abs_d: float | None
    p95_abs_d: float | None
    p99_abs_d: float | None
    threshold_sweep: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_total_cells": self.expected_total_cells,
            "present_cells": self.present_cells,
            "absent_cells": self.absent_cells,
            "n_computable": self.n_computable,
            "n_none": self.n_none,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_zero": self.n_zero,
            "positive_fraction": self.positive_fraction,
            "max_abs_d": self.max_abs_d,
            "mean_abs_d": self.mean_abs_d,
            "median_abs_d": self.median_abs_d,
            "p95_abs_d": self.p95_abs_d,
            "p99_abs_d": self.p99_abs_d,
            "threshold_sweep": dict(self.threshold_sweep),
        }


@dataclass(frozen=True)
class AsymmetryStatistic:
    """Head-level directional asymmetry. NOT cell-level.

    For each head, compute mean d across all present combos. Classify
    as negative/positive/zero using sign_eps deadzone. Heads with
    fewer than min_present_combos are excluded from sign counts.
    """

    n_negative_heads: int
    n_positive_heads: int
    n_zero_heads: int
    n_absent_heads: int  # zero present combos
    n_sparse_heads: int  # 1 to min_combos-1 present combos (excluded from vote)
    signed_excess: int  # n_negative_heads - n_positive_heads
    negative_fraction: float | None  # n_neg / (n_neg + n_pos), None if both zero
    mean_head_mean_d: float | None  # mean of per-head mean-d values (voting heads only)
    mean_head_abs_mean_d: float | None
    min_present_combos: int  # frozen threshold (default 6)
    sign_eps: float  # frozen deadzone (default 1e-10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_negative_heads": self.n_negative_heads,
            "n_positive_heads": self.n_positive_heads,
            "n_zero_heads": self.n_zero_heads,
            "n_absent_heads": self.n_absent_heads,
            "n_sparse_heads": self.n_sparse_heads,
            "signed_excess": self.signed_excess,
            "negative_fraction": self.negative_fraction,
            "mean_head_mean_d": self.mean_head_mean_d,
            "mean_head_abs_mean_d": self.mean_head_abs_mean_d,
            "min_present_combos": self.min_present_combos,
            "sign_eps": self.sign_eps,
        }


@dataclass(frozen=True)
class SubsetSpec:
    """Common output from matching and selection modules."""

    included_indices: frozenset[int]
    provenance_name: str  # "length_matched", "unanimous_only"
    n_correct: int
    n_incorrect: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "included_indices": sorted(self.included_indices),
            "provenance_name": self.provenance_name,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
        }


@dataclass(frozen=True)
class MatchingDiagnostics:
    """Diagnostics from greedy nearest-neighbor length matching."""

    n_correct_before: int
    n_incorrect_before: int
    n_correct_after: int
    n_incorrect_after: int
    n_correct_dropped: int
    n_incorrect_dropped: int
    mean_tokens_correct_before: float
    mean_tokens_incorrect_before: float
    mean_tokens_correct_after: float
    mean_tokens_incorrect_after: float
    max_pair_token_gap: int
    mean_pair_token_gap: float
    dropped_correct_token_summary: str  # "min-max, mean" of dropped correct tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_correct_before": self.n_correct_before,
            "n_incorrect_before": self.n_incorrect_before,
            "n_correct_after": self.n_correct_after,
            "n_incorrect_after": self.n_incorrect_after,
            "n_correct_dropped": self.n_correct_dropped,
            "n_incorrect_dropped": self.n_incorrect_dropped,
            "mean_tokens_correct_before": self.mean_tokens_correct_before,
            "mean_tokens_incorrect_before": self.mean_tokens_incorrect_before,
            "mean_tokens_correct_after": self.mean_tokens_correct_after,
            "mean_tokens_incorrect_after": self.mean_tokens_incorrect_after,
            "max_pair_token_gap": self.max_pair_token_gap,
            "mean_pair_token_gap": self.mean_pair_token_gap,
            "dropped_correct_token_summary": self.dropped_correct_token_summary,
        }


@dataclass(frozen=True)
class SelectionDiagnostics:
    """Diagnostics from cohort selection (e.g. unanimous-only)."""

    selection_name: str
    n_correct_before: int
    n_incorrect_before: int
    n_correct_after: int
    n_incorrect_after: int
    n_excluded_ambiguous: int
    n_excluded_non_unanimous: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_name": self.selection_name,
            "n_correct_before": self.n_correct_before,
            "n_incorrect_before": self.n_incorrect_before,
            "n_correct_after": self.n_correct_after,
            "n_incorrect_after": self.n_incorrect_after,
            "n_excluded_ambiguous": self.n_excluded_ambiguous,
            "n_excluded_non_unanimous": self.n_excluded_non_unanimous,
        }


@dataclass(frozen=True)
class AsymmetryNullResult:
    """Result of a permutation null test on the asymmetry statistic."""

    observed: AsymmetryStatistic
    p_value_two_sided: float  # PRIMARY
    p_value_one_sided_negative: float  # secondary/descriptive
    null_signed_excess_summary: dict[str, float]  # mean, std, min, max, percentiles
    n_permutations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed.to_dict(),
            "p_value_two_sided": self.p_value_two_sided,
            "p_value_one_sided_negative": self.p_value_one_sided_negative,
            "null_signed_excess_summary": dict(self.null_signed_excess_summary),
            "n_permutations": self.n_permutations,
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
    d_landscape: DLandscape | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "eligibility": self.eligibility.to_dict() if self.eligibility is not None else None,
            "observed": self.observed.to_dict(),
            "observed_profile": self.observed_profile.to_dict(),
            "null_at_min_combos": self.null_at_min_combos.to_dict(),
            "null_at_seven": self.null_at_seven.to_dict(),
            "bin_boundaries": list(self.bin_boundaries),
            "bin_counts": {k: dict(v) for k, v in self.bin_counts.items()},
            "d_landscape": self.d_landscape.to_dict() if self.d_landscape is not None else None,
        }
