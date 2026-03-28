"""Stratified permutation null test for PE recurrence.

The only analysis module with RNG. Handles: stratified label
shuffling, null loop, empirical p-values (Phipson-Smyth).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from navi_sad.analysis.recurrence import PELookup, compute_recurrence
from navi_sad.analysis.types import (
    PermutationNullConfig,
    PermutationNullResult,
    RecurrenceNullReport,
)


def assign_length_bins(
    token_counts: dict[int, int],
    labels: dict[int, str],
    n_bins: int = 2,
) -> tuple[dict[int, int], list[int]]:
    """Assign samples to coarse generation-length bins.

    Args:
        token_counts: dataset_index -> generated_token_count.
        labels: dataset_index -> "correct" or "incorrect".
        n_bins: Number of bins. 1 = unstratified. 2 = median split.

    Returns:
        (bin_assignments, bin_boundaries).
        bin_assignments: dataset_index -> bin index (0-based).
        bin_boundaries: sorted list of split points. Boundaries are
            exclusive lower bounds of the upper bin: a sample with
            count == boundary goes to the upper bin (strict < comparison).

    Raises:
        ValueError: If input is empty or a bin has no samples of either class.
    """
    if not token_counts:
        raise ValueError("Cannot assign length bins: empty input")

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if n_bins == 1:
        return dict.fromkeys(token_counts, 0), []

    sorted_counts = sorted(token_counts.values())
    boundaries: list[int] = []
    for i in range(1, n_bins):
        # Quantile-based boundaries
        pos = int(len(sorted_counts) * i / n_bins)
        boundaries.append(sorted_counts[pos])

    def _get_bin(count: int) -> int:
        for i, boundary in enumerate(boundaries):
            if count < boundary:
                return i
        return len(boundaries)

    assignments = {idx: _get_bin(count) for idx, count in token_counts.items()}

    # Validate: every expected bin must have >= 1 sample of each class.
    # Check all bins 0..n_bins-1, not just populated ones.
    bin_class_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idx, bin_id in assignments.items():
        bin_class_counts[bin_id][labels[idx]] += 1

    for bin_id in range(n_bins):
        for cls in ("correct", "incorrect"):
            if bin_class_counts[bin_id].get(cls, 0) == 0:
                raise ValueError(
                    f"Bin {bin_id} has no {cls!r} samples. "
                    f"Stratification not possible with {n_bins} bins. "
                    f"Bin contents: {dict(bin_class_counts[bin_id])}"
                )

    return assignments, boundaries


def stratified_permute_labels(
    labels: dict[int, str],
    bins: dict[int, int],
    rng: random.Random,
) -> dict[int, str]:
    """Shuffle labels within each bin, preserving class sizes.

    Args:
        labels: dataset_index -> "correct" or "incorrect".
        bins: dataset_index -> bin index.
        rng: Random instance for reproducibility.

    Returns:
        New label assignment with same class sizes within each bin.
    """
    # Group indices by bin
    bin_indices: dict[int, list[int]] = defaultdict(list)
    for idx in labels:
        if idx not in bins:
            raise ValueError(f"Sample {idx} has a label but no bin assignment")
        bin_indices[bins[idx]].append(idx)

    shuffled: dict[int, str] = {}
    for bin_id in sorted(bin_indices):
        indices = bin_indices[bin_id]
        bin_labels = [labels[idx] for idx in indices]
        rng.shuffle(bin_labels)
        for idx, label in zip(indices, bin_labels, strict=True):
            shuffled[idx] = label

    return shuffled


def compute_null_result(
    observed: int,
    null_counts: list[int],
) -> PermutationNullResult:
    """Compute null summary statistics from permutation distribution.

    p-value uses Phipson-Smyth correction: (k + 1) / (N + 1)
    where k = count of null values >= observed.

    Raises:
        ValueError: If null_counts is empty (no permutations were run).
    """
    if not null_counts:
        raise ValueError(
            "null_counts is empty — no permutations were run. "
            "Cannot compute a p-value from zero permutations."
        )
    n = len(null_counts)
    k = sum(1 for nc in null_counts if nc >= observed)
    p_value = (k + 1) / (n + 1)

    mean_val = sum(null_counts) / n if n > 0 else 0.0
    # Population std: null_counts is the complete discrete distribution, not a sample.
    variance = sum((x - mean_val) ** 2 for x in null_counts) / n if n > 0 else 0.0
    std_val = math.sqrt(variance)

    sorted_counts = sorted(null_counts)
    percentiles: dict[int, int] = {}
    for pct in (5, 25, 50, 75, 95):
        idx = int(n * pct / 100)
        idx = min(idx, n - 1)
        percentiles[pct] = sorted_counts[idx] if n > 0 else 0

    return PermutationNullResult(
        observed=observed,
        null_counts=null_counts,
        p_value=p_value,
        expected_under_null=mean_val,
        null_mean=mean_val,
        null_std=std_val,
        null_min=min(null_counts) if null_counts else 0,
        null_max=max(null_counts) if null_counts else 0,
        null_percentiles=percentiles,
    )


def run_permutation_null(
    lookup: PELookup,
    labels: dict[int, str],
    token_counts: dict[int, int],
    *,
    config: PermutationNullConfig,
    num_layers: int,
    num_heads: int,
) -> RecurrenceNullReport:
    """Run the full stratified permutation null test.

    1. Compute observed recurrence under true labels.
    2. Assign length bins and validate strata.
    3. Loop N permutations: shuffle labels within bins, recompute recurrence.
    4. Compute p-values at min_combos and at 7.

    Args:
        lookup: Precomputed PE lookup table.
        labels: True label assignments.
        token_counts: dataset_index -> generated_token_count.
        config: Null test configuration.
        num_layers: Number of model layers.
        num_heads: Number of attention heads per layer.

    Returns:
        RecurrenceNullReport with all results.
    """
    # Observed statistic
    observed_stat, observed_profile = compute_recurrence(
        lookup,
        labels,
        d_threshold=config.d_threshold,
        min_combos=config.min_combos,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Observed count at level 7
    observed_at_seven = observed_profile.counts_at_level.get(7, 0)

    # Stratification
    bins, bin_boundaries = assign_length_bins(
        token_counts,
        labels,
        n_bins=config.n_bins,
    )

    # Bin counts for provenance
    bin_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for idx, bin_id in bins.items():
        bin_counts[str(bin_id)][labels[idx]] += 1

    # Permutation loop
    rng = random.Random(config.seed)
    null_at_min: list[int] = []
    null_at_seven: list[int] = []

    for _ in range(config.n_permutations):
        shuffled = stratified_permute_labels(labels, bins, rng)
        perm_stat, perm_profile = compute_recurrence(
            lookup,
            shuffled,
            d_threshold=config.d_threshold,
            min_combos=config.min_combos,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        null_at_min.append(perm_stat.recurring_head_count)
        null_at_seven.append(perm_profile.counts_at_level.get(7, 0))

    return RecurrenceNullReport(
        config=config,
        eligibility=None,  # Caller constructs a new report with eligibility attached
        observed=observed_stat,
        observed_profile=observed_profile,
        null_at_min_combos=compute_null_result(
            observed_stat.recurring_head_count,
            null_at_min,
        ),
        null_at_seven=compute_null_result(observed_at_seven, null_at_seven),
        bin_boundaries=bin_boundaries,
        bin_counts={k: dict(v) for k, v in bin_counts.items()},
    )
