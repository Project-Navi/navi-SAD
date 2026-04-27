"""Greedy nearest-neighbor length matching for confound control.

Pure deterministic matching. No RNG. Iteration order: incorrect
samples in ascending dataset_index. Ties broken by smallest
correct dataset_index.
"""

from __future__ import annotations

import structlog

from navi_sad.analysis.types import MatchingDiagnostics, SubsetSpec

log = structlog.get_logger()


def match_by_token_count(
    labels: dict[int, str],
    token_counts: dict[int, int],
) -> tuple[SubsetSpec, MatchingDiagnostics, list[tuple[int, int]]]:
    """Match incorrect samples to correct samples by token count.

    Greedy nearest-neighbor without replacement. Each incorrect
    sample matched to the closest available correct sample.

    Args:
        labels: dataset_index -> "correct" or "incorrect".
        token_counts: dataset_index -> generated_token_count.

    Returns:
        (SubsetSpec, MatchingDiagnostics, pairs) where pairs is a
        list of (correct_idx, incorrect_idx) tuples.
    """
    correct_indices = sorted(idx for idx, lab in labels.items() if lab == "correct")
    incorrect_indices = sorted(idx for idx, lab in labels.items() if lab == "incorrect")

    n_correct_before = len(correct_indices)
    n_incorrect_before = len(incorrect_indices)

    # Pre-match token stats
    mean_correct_before = (
        sum(token_counts[i] for i in correct_indices) / n_correct_before
        if n_correct_before
        else 0.0
    )
    mean_incorrect_before = (
        sum(token_counts[i] for i in incorrect_indices) / n_incorrect_before
        if n_incorrect_before
        else 0.0
    )

    # Greedy matching: iterate incorrect in ascending order
    available_correct = set(correct_indices)
    pairs: list[tuple[int, int]] = []
    matched_correct: list[int] = []
    matched_incorrect: list[int] = []

    for inc_idx in incorrect_indices:
        if not available_correct:
            break
        inc_tokens = token_counts[inc_idx]
        # Find closest correct by token count, tie-break by smallest index
        best = min(
            available_correct,
            key=lambda c: (abs(token_counts[c] - inc_tokens), c),
        )
        available_correct.remove(best)
        pairs.append((best, inc_idx))
        matched_correct.append(best)
        matched_incorrect.append(inc_idx)

    # Compute diagnostics
    n_correct_after = len(matched_correct)
    n_incorrect_after = len(matched_incorrect)
    n_correct_dropped = n_correct_before - n_correct_after
    n_incorrect_dropped = n_incorrect_before - n_incorrect_after

    mean_correct_after = (
        sum(token_counts[i] for i in matched_correct) / n_correct_after if n_correct_after else 0.0
    )
    mean_incorrect_after = (
        sum(token_counts[i] for i in matched_incorrect) / n_incorrect_after
        if n_incorrect_after
        else 0.0
    )

    pair_gaps = [abs(token_counts[c] - token_counts[i]) for c, i in pairs]
    max_gap = max(pair_gaps) if pair_gaps else 0
    mean_gap = sum(pair_gaps) / len(pair_gaps) if pair_gaps else 0.0

    dropped_correct = sorted(available_correct)
    if dropped_correct:
        dropped_tokens = [token_counts[i] for i in dropped_correct]
        dropped_summary = (
            f"{min(dropped_tokens)}-{max(dropped_tokens)}, "
            f"mean={sum(dropped_tokens) / len(dropped_tokens):.1f}"
        )
    else:
        dropped_summary = "none"

    included = frozenset(matched_correct + matched_incorrect)

    spec = SubsetSpec(
        included_indices=included,
        provenance_name="length_matched",
        n_correct=n_correct_after,
        n_incorrect=n_incorrect_after,
    )

    diag = MatchingDiagnostics(
        n_correct_before=n_correct_before,
        n_incorrect_before=n_incorrect_before,
        n_correct_after=n_correct_after,
        n_incorrect_after=n_incorrect_after,
        n_correct_dropped=n_correct_dropped,
        n_incorrect_dropped=n_incorrect_dropped,
        mean_tokens_correct_before=mean_correct_before,
        mean_tokens_incorrect_before=mean_incorrect_before,
        mean_tokens_correct_after=mean_correct_after,
        mean_tokens_incorrect_after=mean_incorrect_after,
        max_pair_token_gap=max_gap,
        mean_pair_token_gap=mean_gap,
        dropped_correct_token_summary=dropped_summary,
    )

    log.info(
        "length_matching_complete",
        n_pairs=len(pairs),
        n_correct_dropped=n_correct_dropped,
        n_incorrect_dropped=n_incorrect_dropped,
        max_gap=max_gap,
        mean_gap=mean_gap,
    )

    return spec, diag, pairs
