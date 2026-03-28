"""Per-head recurrence statistic from PE features.

Computes how many (mode, segment) combinations each head exceeds
the Cohen's d threshold. Defines the frozen test statistic.
No RNG. No label shuffling. Pure deterministic computation.
"""

from __future__ import annotations

from navi_sad.analysis.types import RecurrenceProfile, RecurrenceStatistic
from navi_sad.pilot.helpers import compute_cohens_d
from navi_sad.signal.pe_features import SamplePEFeatures

# Type alias for the PE lookup table.
# Outer: (mode, segment) -> inner: (layer, head) -> {dataset_index: pe_value}
PELookup = dict[tuple[str, str], dict[tuple[int, int], dict[int, float]]]


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
        correct_vals = [v for idx, v in pe_by_idx.items() if labels[idx] == "correct"]
        incorrect_vals = [v for idx, v in pe_by_idx.items() if labels[idx] == "incorrect"]
        d_val, _ = compute_cohens_d(correct_vals, incorrect_vals)
        result[head_key] = d_val
    return result


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

    For each (layer, head), counts how many (mode, segment) combos
    have |Cohen's d| strictly greater than d_threshold.

    Returns:
        (RecurrenceStatistic, RecurrenceProfile) tuple.
    """
    # Initialize all heads to 0
    combo_counts: dict[tuple[int, int], int] = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            combo_counts[(layer, head)] = 0

    # For each combo, compute d values and tally
    for _combo_key, head_pe in lookup.items():
        d_values = compute_combo_cohens_d(head_pe, labels)
        for head_key, d_val in d_values.items():
            if d_val is not None and abs(d_val) > d_threshold:
                if head_key in combo_counts:
                    combo_counts[head_key] += 1

    total_heads = num_layers * num_heads
    recurring = sum(1 for v in combo_counts.values() if v >= min_combos)

    # Build profile: at each level, how many heads have >= that count
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
