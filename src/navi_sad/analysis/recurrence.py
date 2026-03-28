"""Per-head recurrence statistic from PE features.

Computes the full Cohen's d matrix across all (mode, segment) x (layer, head)
combinations, then derives recurrence counts, threshold sweeps, and directional
summaries from it. No RNG. No label shuffling. Pure deterministic computation.

Uses numpy for vectorized computation. The pure-Python compute_cohens_d in
stats/effect_size.py is kept for single-pair use; this module operates on
the full grid.
"""

from __future__ import annotations

import numpy as np

from navi_sad.analysis.types import RecurrenceProfile, RecurrenceStatistic
from navi_sad.signal.pe_features import SamplePEFeatures

# Type alias for the PE lookup table.
# Outer: (mode, segment) -> inner: (layer, head) -> {dataset_index: pe_value}
PELookup = dict[tuple[str, str], dict[tuple[int, int], dict[int, float]]]

CANONICAL_LABELS = frozenset({"correct", "incorrect"})

# Frozen contract: 3 modes x 4 segments = 12 combos per head.
EXPECTED_COMBOS: frozenset[tuple[str, str]] = frozenset(
    (mode, segment)
    for mode in ("raw", "diff", "residual")
    for segment in ("full", "early", "mid", "late")
)

# Type alias for the full d-value matrix.
# (mode, segment) -> (layer, head) -> d_value or None
DMatrix = dict[tuple[str, str], dict[tuple[int, int], float | None]]


def build_pe_lookup(
    samples: dict[int, SamplePEFeatures],
) -> PELookup:
    """Build flat PE lookup from SamplePEFeatures.

    Indexes PE values by (mode, segment) -> (layer, head) -> {dataset_index: pe}.
    Only includes entries where eligible=True AND pe is not None.
    """
    lookup: PELookup = {}
    for idx, pe_features in samples.items():
        for h in pe_features.heads:
            if not h.eligible or h.pe is None:
                continue
            combo = (h.mode, h.segment)
            if combo not in lookup:
                lookup[combo] = {}
            head_key = (h.layer_idx, h.head_idx)
            if head_key not in lookup[combo]:
                lookup[combo][head_key] = {}
            lookup[combo][head_key][idx] = h.pe

    return lookup


def validate_combo_set(
    lookup: PELookup,
    *,
    expected: frozenset[tuple[str, str]] = EXPECTED_COMBOS,
) -> None:
    """Validate that the PE lookup contains exactly the expected combos.

    Raises:
        ValueError: If combos in lookup do not match expected set.
            Reports missing and unexpected combos.
    """
    actual = frozenset(lookup.keys())
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing combos: {sorted(missing)}")
        if unexpected:
            parts.append(f"unexpected combos: {sorted(unexpected)}")
        raise ValueError(
            f"PE lookup combo set does not match expected 12-combo contract. {'; '.join(parts)}"
        )


_POOLED_VAR_EPS = 1e-12


def _cohens_d_vectorized(
    correct_vals: np.ndarray,
    incorrect_vals: np.ndarray,
) -> float | None:
    """Compute Cohen's d using numpy. Returns None if insufficient data or degenerate."""
    n_a = len(correct_vals)
    n_b = len(incorrect_vals)
    if n_a < 2 or n_b < 2:
        return None
    mean_a = correct_vals.mean()
    mean_b = incorrect_vals.mean()
    var_a = correct_vals.var(ddof=1)
    var_b = incorrect_vals.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= _POOLED_VAR_EPS:
        return None
    return float((mean_a - mean_b) / np.sqrt(pooled_var))


def compute_combo_cohens_d(
    head_pe: dict[tuple[int, int], dict[int, float]],
    labels: dict[int, str],
) -> dict[tuple[int, int], float | None]:
    """Compute Cohen's d for every head in one (mode, segment) combo.

    Args:
        head_pe: Mapping (layer, head) -> {dataset_index: pe_value}.
        labels: Mapping dataset_index -> "correct" or "incorrect".

    Returns:
        Dict (layer, head) -> d_value or None if insufficient data.
    """
    result: dict[tuple[int, int], float | None] = {}
    for head_key, pe_by_idx in head_pe.items():
        correct_vals = np.array([v for idx, v in pe_by_idx.items() if labels[idx] == "correct"])
        incorrect_vals = np.array([v for idx, v in pe_by_idx.items() if labels[idx] == "incorrect"])
        result[head_key] = _cohens_d_vectorized(correct_vals, incorrect_vals)
    return result


def compute_d_matrix(
    lookup: PELookup,
    labels: dict[int, str],
    *,
    num_layers: int,
    num_heads: int,
) -> DMatrix:
    """Compute the full Cohen's d matrix across all combos and heads.

    Returns the raw d values for every (combo, head) pair. This is
    the foundation for recurrence, threshold sweeps, and directional
    analysis. Nothing is discarded.

    Raises:
        ValueError: If labels contain non-canonical values or heads
            fall outside the declared grid.
    """
    stray = set(labels.values()) - CANONICAL_LABELS
    if stray:
        raise ValueError(
            f"Labels contain non-canonical values: {sorted(stray)}. "
            f"Only {sorted(CANONICAL_LABELS)} are accepted."
        )
    if num_layers < 1 or num_heads < 1:
        raise ValueError(
            f"num_layers and num_heads must be >= 1, "
            f"got num_layers={num_layers}, num_heads={num_heads}"
        )

    grid = frozenset((layer, head) for layer in range(num_layers) for head in range(num_heads))
    d_matrix: DMatrix = {}

    for combo_key, head_pe in lookup.items():
        d_values = compute_combo_cohens_d(head_pe, labels)
        for head_key in d_values:
            if head_key not in grid:
                raise ValueError(
                    f"Head {head_key} in PE lookup is outside the declared "
                    f"grid ({num_layers} layers x {num_heads} heads). "
                    f"Check that num_layers and num_heads match the data."
                )
        d_matrix[combo_key] = d_values

    return d_matrix


def recurrence_from_d_matrix(
    d_matrix: DMatrix,
    *,
    d_threshold: float,
    min_combos: int,
    num_layers: int,
    num_heads: int,
) -> tuple[RecurrenceStatistic, RecurrenceProfile]:
    """Compute recurrence statistic from a pre-computed d matrix.

    For each (layer, head), counts how many combos have |d| strictly
    greater than d_threshold. This is a pure reduction over the d matrix.
    """
    combo_counts: dict[tuple[int, int], int] = {
        (layer, head): 0 for layer in range(num_layers) for head in range(num_heads)
    }

    for _combo_key, head_d in d_matrix.items():
        for head_key, d_val in head_d.items():
            if d_val is not None and abs(d_val) > d_threshold:
                if head_key in combo_counts:
                    combo_counts[head_key] += 1

    total_heads = num_layers * num_heads
    recurring = sum(1 for v in combo_counts.values() if v >= min_combos)

    max_possible = max(combo_counts.values()) if combo_counts else 0
    counts_at_level: dict[int, int] = {}
    for level in range(1, max(max_possible, 12) + 1):
        counts_at_level[level] = sum(1 for v in combo_counts.values() if v >= level)

    return (
        RecurrenceStatistic(
            d_threshold=d_threshold,
            min_combos=min_combos,
            recurring_head_count=recurring,
            total_heads=total_heads,
            per_head_combo_counts=combo_counts,
        ),
        RecurrenceProfile(counts_at_level=counts_at_level),
    )


def compute_recurrence(
    lookup: PELookup,
    labels: dict[int, str],
    *,
    d_threshold: float,
    min_combos: int,
    num_layers: int,
    num_heads: int,
) -> tuple[RecurrenceStatistic, RecurrenceProfile]:
    """Compute recurrence statistic across all combos.

    Convenience wrapper: computes d matrix then reduces to recurrence.
    If you need the d values for downstream analysis (threshold sweeps,
    directional analysis), call compute_d_matrix() directly.
    """
    d_matrix = compute_d_matrix(lookup, labels, num_layers=num_layers, num_heads=num_heads)
    return recurrence_from_d_matrix(
        d_matrix,
        d_threshold=d_threshold,
        min_combos=min_combos,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def summarize_d_matrix(d_matrix: DMatrix) -> dict[str, object]:
    """Compute summary statistics from a d matrix.

    Returns landscape characterization: distribution of |d|, directional
    counts, and a threshold sweep showing how many (head, combo) cells
    exceed each threshold.
    """
    all_d: list[float] = []
    n_none = 0

    for _combo, head_d in d_matrix.items():
        for _head, d_val in head_d.items():
            if d_val is None:
                n_none += 1
            else:
                all_d.append(d_val)

    if not all_d:
        return {
            "n_total": n_none,
            "n_computable": 0,
            "n_none": n_none,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "positive_fraction": None,
            "max_abs_d": None,
            "mean_abs_d": None,
            "median_abs_d": None,
            "p95_abs_d": None,
            "p99_abs_d": None,
            "threshold_sweep": {},
        }

    d_arr = np.array(all_d)
    abs_d = np.abs(d_arr)
    n_positive = int(np.sum(d_arr > 0))
    n_negative = int(np.sum(d_arr < 0))
    n_zero = int(np.sum(d_arr == 0))
    n_signed = n_positive + n_negative

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

    return {
        "n_total": len(all_d) + n_none,
        "n_computable": len(all_d),
        "n_none": n_none,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "positive_fraction": n_positive / n_signed if n_signed > 0 else None,
        "max_abs_d": float(abs_d.max()),
        "mean_abs_d": float(abs_d.mean()),
        "median_abs_d": float(np.median(abs_d)),
        "p95_abs_d": float(np.percentile(abs_d, 95)),
        "p99_abs_d": float(np.percentile(abs_d, 99)),
        "threshold_sweep": {str(t): int(np.sum(abs_d > t)) for t in thresholds},
    }
