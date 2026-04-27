"""Stratified permutation null test for PE recurrence.

The only analysis module with RNG. Handles: stratified label
shuffling, null loop, empirical p-values (Phipson-Smyth).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import structlog

from navi_sad.analysis.recurrence import (
    PELookup,
    compute_d_matrix,
    compute_head_asymmetry,
    compute_recurrence,
)
from navi_sad.analysis.types import (
    AsymmetryNullResult,
    NullDistributionSummary,
    PermutationNullConfig,
    PermutationNullResult,
    RecurrenceNullReport,
)

log = structlog.get_logger()


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
        indices = sorted(bin_indices[bin_id])  # Sort for insertion-order independence
        bin_labels = [labels[idx] for idx in indices]
        rng.shuffle(bin_labels)
        for idx, label in zip(indices, bin_labels, strict=True):
            shuffled[idx] = label

    return shuffled


def _compute_distribution_stats(values: list[int]) -> NullDistributionSummary:
    """Compute descriptive summary of a null distribution.

    Shared implementation for both recurrence null and asymmetry null.
    Population std (ddof=0). Nearest-rank percentiles (no interpolation).
    """
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / n
    std_val = math.sqrt(variance)

    sorted_vals = sorted(values)
    percentiles: dict[int, float] = {}
    for pct in (5, 25, 50, 75, 95):
        idx = min(int(n * pct / 100), n - 1)
        percentiles[pct] = float(sorted_vals[idx])

    return NullDistributionSummary(
        mean=mean_val,
        std=std_val,
        min_val=float(min(values)),
        max_val=float(max(values)),
        percentiles=percentiles,
        n=n,
    )


_VALID_TAILS = frozenset({"right", "left", "two-sided"})


def compute_null_result(
    observed: int,
    null_counts: list[int],
    tail: str = "right",
) -> PermutationNullResult:
    """Compute null summary statistics from permutation distribution.

    p-value uses Phipson-Smyth correction: (k + 1) / (N + 1).

    Args:
        observed: Observed test statistic.
        null_counts: Null distribution values.
        tail: Which tail to test.
            "right": k = count(null >= observed). Default.
            "left": k = count(null <= observed).
            "two-sided": k = count(|null| >= |observed|).

    Raises:
        ValueError: If null_counts is empty or tail is invalid.
    """
    if tail not in _VALID_TAILS:
        raise ValueError(f"tail must be one of {sorted(_VALID_TAILS)}, got {tail!r}")
    if not null_counts:
        raise ValueError(
            "null_counts is empty — no permutations were run. "
            "Cannot compute a p-value from zero permutations."
        )
    n = len(null_counts)
    if tail == "right":
        k = sum(1 for nc in null_counts if nc >= observed)
    elif tail == "left":
        k = sum(1 for nc in null_counts if nc <= observed)
    else:  # two-sided
        abs_obs = abs(observed)
        k = sum(1 for nc in null_counts if abs(nc) >= abs_obs)
    p_value = (k + 1) / (n + 1)

    # Shared summary computation — same semantics as asymmetry null path.
    summary = _compute_distribution_stats(null_counts)

    # Populate flat fields from summary (keep PermutationNullResult's public shape).
    return PermutationNullResult(
        observed=observed,
        null_counts=null_counts,
        p_value=p_value,
        expected_under_null=summary.mean,
        null_mean=summary.mean,
        null_std=summary.std,
        null_min=int(summary.min_val),
        null_max=int(summary.max_val),
        null_percentiles={k: int(v) for k, v in summary.percentiles.items()},
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
    log.info(
        "permutation_null_started",
        n_permutations=config.n_permutations,
        seed=config.seed,
        n_bins=config.n_bins,
        d_threshold=config.d_threshold,
    )

    # Observed statistic
    observed_stat, observed_profile = compute_recurrence(
        lookup,
        labels,
        d_threshold=config.d_threshold,
        min_combos=config.min_combos,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    log.info(
        "observed_recurrence_computed",
        n_heads=num_layers * num_heads,
        recurring_head_count=observed_stat.recurring_head_count,
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

    null_result_min = compute_null_result(
        observed_stat.recurring_head_count,
        null_at_min,
    )
    null_result_seven = compute_null_result(observed_at_seven, null_at_seven)

    log.info(
        "permutation_null_complete",
        observed_at_min=observed_stat.recurring_head_count,
        p_value_at_min=null_result_min.p_value,
        observed_at_seven=observed_at_seven,
        p_value_at_seven=null_result_seven.p_value,
    )

    return RecurrenceNullReport(
        config=config,
        eligibility=None,  # Caller constructs a new report with eligibility attached
        observed=observed_stat,
        observed_profile=observed_profile,
        null_at_min_combos=null_result_min,
        null_at_seven=null_result_seven,
        bin_boundaries=bin_boundaries,
        bin_counts={k: dict(v) for k, v in bin_counts.items()},
    )


# -- Asymmetry null (PR #30) --


def run_asymmetry_null(
    lookup: PELookup,
    labels: dict[int, str],
    token_counts: dict[int, int],
    *,
    num_layers: int,
    num_heads: int,
    n_permutations: int = 10_000,
    n_bins: int = 2,
    seed: int = 42,
    min_present_combos: int = 6,
    sign_eps: float = 1e-10,
) -> AsymmetryNullResult:
    """Stratified permutation null for the head-level asymmetry statistic.

    Each permutation: shuffle labels within length bins, recompute
    d matrix, compute head asymmetry, record signed_excess.

    Returns AsymmetryNullResult with two-sided (primary) and
    one-sided (secondary/descriptive) p-values.
    """
    log.info(
        "asymmetry_null_started",
        n_permutations=n_permutations,
        seed=seed,
        n_bins=n_bins,
    )

    # Observed
    observed_d_matrix = compute_d_matrix(lookup, labels, num_layers=num_layers, num_heads=num_heads)
    observed_stat = compute_head_asymmetry(
        observed_d_matrix,
        num_layers=num_layers,
        num_heads=num_heads,
        min_present_combos=min_present_combos,
        sign_eps=sign_eps,
    )

    n_computable = sum(1 for hd in observed_d_matrix.values() for d in hd.values() if d is not None)
    log.info(
        "observed_asymmetry_computed",
        n_combos=len(observed_d_matrix),
        n_heads=num_layers * num_heads,
        n_computable=n_computable,
        n_voting=(
            observed_stat.n_negative_heads
            + observed_stat.n_positive_heads
            + observed_stat.n_zero_heads
        ),
        n_absent=observed_stat.n_absent_heads,
        n_sparse=observed_stat.n_sparse_heads,
        signed_excess=observed_stat.signed_excess,
    )

    # Stratification
    bins, _boundaries = assign_length_bins(token_counts, labels, n_bins=n_bins)

    # Permutation loop
    rng = random.Random(seed)
    null_signed_excesses: list[int] = []

    for _ in range(n_permutations):
        shuffled = stratified_permute_labels(labels, bins, rng)
        perm_d_matrix = compute_d_matrix(
            lookup, shuffled, num_layers=num_layers, num_heads=num_heads
        )
        perm_stat = compute_head_asymmetry(
            perm_d_matrix,
            num_layers=num_layers,
            num_heads=num_heads,
            min_present_combos=min_present_combos,
            sign_eps=sign_eps,
        )
        null_signed_excesses.append(perm_stat.signed_excess)

    # Two-sided: |null| >= |observed|
    obs_se = observed_stat.signed_excess
    abs_obs = abs(obs_se)
    k_two = sum(1 for nc in null_signed_excesses if abs(nc) >= abs_obs)
    p_two_sided = (k_two + 1) / (n_permutations + 1)

    # One-sided negative: tests whether negative direction is more
    # extreme than expected. signed_excess = n_neg - n_pos, so large
    # positive = more negative heads. Right-tail: null >= observed.
    k_neg = sum(1 for nc in null_signed_excesses if nc >= obs_se)
    p_one_sided_negative = (k_neg + 1) / (n_permutations + 1)

    log.info(
        "asymmetry_null_complete",
        signed_excess=obs_se,
        p_two_sided=p_two_sided,
        p_one_sided_negative=p_one_sided_negative,
    )

    return AsymmetryNullResult(
        observed=observed_stat,
        p_value_two_sided=p_two_sided,
        p_value_one_sided_negative=p_one_sided_negative,
        null_signed_excess_summary=_compute_distribution_stats(null_signed_excesses),
        n_permutations=n_permutations,
    )


def _paired_permute_labels(
    labels: dict[int, str],
    pairs: list[tuple[int, int]],
    rng: random.Random,
) -> dict[int, str]:
    """Within-pair label swap: each pair independently swaps or keeps.

    pairs: list of (correct_idx, incorrect_idx) tuples.
    For each pair, with probability 0.5, swap the labels.
    """
    shuffled = dict(labels)
    for idx_a, idx_b in pairs:
        if rng.random() < 0.5:
            shuffled[idx_a], shuffled[idx_b] = shuffled[idx_b], shuffled[idx_a]
    return shuffled


def run_paired_asymmetry_null(
    lookup: PELookup,
    labels: dict[int, str],
    pairs: list[tuple[int, int]],
    *,
    num_layers: int,
    num_heads: int,
    n_permutations: int = 10_000,
    seed: int = 42,
    min_present_combos: int = 6,
    sign_eps: float = 1e-10,
) -> AsymmetryNullResult:
    """Pair-restricted permutation null for matched designs.

    Each pair is a block. Under the null, labels are swapped or
    kept independently per pair (coin flip). This preserves the
    matching structure exactly.

    Args:
        lookup: Precomputed PE lookup table.
        labels: Label assignments (includes only paired samples).
        pairs: List of (correct_idx, incorrect_idx) tuples.
        num_layers: Number of model layers.
        num_heads: Number of attention heads per layer.
        n_permutations: Number of permutations.
        seed: RNG seed for reproducibility.
        min_present_combos: Minimum combos for a head to vote.
        sign_eps: Deadzone for zero classification.

    Raises:
        ValueError: If pairs is empty.
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")

    log.info(
        "paired_null_started",
        n_pairs=len(pairs),
        n_permutations=n_permutations,
        seed=seed,
    )

    # Observed
    observed_d_matrix = compute_d_matrix(lookup, labels, num_layers=num_layers, num_heads=num_heads)
    observed_stat = compute_head_asymmetry(
        observed_d_matrix,
        num_layers=num_layers,
        num_heads=num_heads,
        min_present_combos=min_present_combos,
        sign_eps=sign_eps,
    )

    n_computable = sum(1 for hd in observed_d_matrix.values() for d in hd.values() if d is not None)
    log.info(
        "observed_paired_asymmetry_computed",
        n_combos=len(observed_d_matrix),
        n_heads=num_layers * num_heads,
        n_computable=n_computable,
        n_voting=(
            observed_stat.n_negative_heads
            + observed_stat.n_positive_heads
            + observed_stat.n_zero_heads
        ),
        n_absent=observed_stat.n_absent_heads,
        n_sparse=observed_stat.n_sparse_heads,
        signed_excess=observed_stat.signed_excess,
    )

    # Permutation loop
    rng = random.Random(seed)
    null_signed_excesses: list[int] = []

    for _ in range(n_permutations):
        shuffled = _paired_permute_labels(labels, pairs, rng)
        perm_d_matrix = compute_d_matrix(
            lookup, shuffled, num_layers=num_layers, num_heads=num_heads
        )
        perm_stat = compute_head_asymmetry(
            perm_d_matrix,
            num_layers=num_layers,
            num_heads=num_heads,
            min_present_combos=min_present_combos,
            sign_eps=sign_eps,
        )
        null_signed_excesses.append(perm_stat.signed_excess)

    # P-values (same logic as run_asymmetry_null)
    obs_se = observed_stat.signed_excess
    abs_obs = abs(obs_se)
    k_two = sum(1 for nc in null_signed_excesses if abs(nc) >= abs_obs)
    p_two_sided = (k_two + 1) / (n_permutations + 1)

    k_neg = sum(1 for nc in null_signed_excesses if nc >= obs_se)
    p_one_sided_negative = (k_neg + 1) / (n_permutations + 1)

    log.info(
        "paired_null_complete",
        signed_excess=obs_se,
        p_two_sided=p_two_sided,
        p_one_sided_negative=p_one_sided_negative,
    )

    return AsymmetryNullResult(
        observed=observed_stat,
        p_value_two_sided=p_two_sided,
        p_value_one_sided_negative=p_one_sided_negative,
        null_signed_excess_summary=_compute_distribution_stats(null_signed_excesses),
        n_permutations=n_permutations,
    )
