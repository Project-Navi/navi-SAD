"""Deterministic cohort selection for confound control.

Pure filtering. No RNG. Produces SubsetSpec with diagnostics.
"""

from __future__ import annotations

from navi_sad.analysis.types import SelectionDiagnostics, SubsetSpec


def select_unanimous(
    reviewer_votes: dict[int, list[str]],
    majority_labels: dict[int, str],
) -> tuple[SubsetSpec, SelectionDiagnostics]:
    """Keep only samples where all reviewers agree on the label.

    Includes unanimous correct and unanimous incorrect. Excludes
    ambiguous majority labels and non-unanimous majority votes.

    Args:
        reviewer_votes: {dataset_index: [reviewer_0_label, ...]}.
        majority_labels: {dataset_index: majority_label}.

    Returns:
        (SubsetSpec, SelectionDiagnostics).
    """
    n_correct_before = sum(1 for v in majority_labels.values() if v == "correct")
    n_incorrect_before = sum(1 for v in majority_labels.values() if v == "incorrect")

    included: list[int] = []
    n_excluded_ambiguous = 0
    n_excluded_non_unanimous = 0

    for idx in sorted(reviewer_votes):
        majority = majority_labels.get(idx)
        if majority not in ("correct", "incorrect"):
            n_excluded_ambiguous += 1
            continue

        votes = reviewer_votes[idx]
        if all(v == votes[0] for v in votes):
            # Unanimous — include
            included.append(idx)
        else:
            n_excluded_non_unanimous += 1

    n_correct_after = sum(1 for i in included if majority_labels[i] == "correct")
    n_incorrect_after = sum(1 for i in included if majority_labels[i] == "incorrect")

    spec = SubsetSpec(
        included_indices=frozenset(included),
        provenance_name="unanimous_only",
        n_correct=n_correct_after,
        n_incorrect=n_incorrect_after,
    )

    diag = SelectionDiagnostics(
        selection_name="unanimous_only",
        n_correct_before=n_correct_before,
        n_incorrect_before=n_incorrect_before,
        n_correct_after=n_correct_after,
        n_incorrect_after=n_incorrect_after,
        n_excluded_ambiguous=n_excluded_ambiguous,
        n_excluded_non_unanimous=n_excluded_non_unanimous,
    )

    return spec, diag
