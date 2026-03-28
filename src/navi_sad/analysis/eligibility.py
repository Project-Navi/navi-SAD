"""Eligibility accounting by class x mode x segment.

Counts how many samples per class have eligible (and pe-present) heads
for each (mode, segment) combination. No statistics. Just accounting.
"""

from __future__ import annotations

from collections import defaultdict

from navi_sad.analysis.types import EligibilityCell, EligibilityTable
from navi_sad.signal.pe_features import SamplePEFeatures

VALID_LABELS = frozenset({"correct", "incorrect"})


def build_eligibility_table(
    samples: dict[int, SamplePEFeatures],
    labels: dict[int, str],
) -> EligibilityTable:
    """Build eligibility table from precomputed PE features.

    Args:
        samples: Mapping of dataset_index -> SamplePEFeatures.
        labels: Mapping of dataset_index -> "correct" or "incorrect".
            Every key in samples must have a label. No other label
            values are accepted.

    Returns:
        EligibilityTable with one cell per (mode, segment) combination.

    Raises:
        ValueError: If any label is not "correct" or "incorrect".
        KeyError: If a sample has no corresponding label.
    """
    if not samples:
        return EligibilityTable(cells=[], n_correct=0, n_incorrect=0)

    # Validate labels
    for idx in samples:
        label = labels[idx]  # KeyError if missing
        if label not in VALID_LABELS:
            raise ValueError(
                f"Sample {idx}: unknown label {label!r}. Must be one of: {sorted(VALID_LABELS)}"
            )

    n_correct = sum(1 for idx in samples if labels[idx] == "correct")
    n_incorrect = sum(1 for idx in samples if labels[idx] == "incorrect")

    # For each (mode, segment), track which samples are eligible,
    # which have pe != None (pe-present), and total. A sample is
    # eligible for a combo if ANY of its heads in that combo are eligible.
    # A sample is pe-present if ANY eligible head has pe != None.
    combo_correct_eligible: dict[tuple[str, str], set[int]] = defaultdict(set)
    combo_incorrect_eligible: dict[tuple[str, str], set[int]] = defaultdict(set)
    combo_correct_pe_present: dict[tuple[str, str], set[int]] = defaultdict(set)
    combo_incorrect_pe_present: dict[tuple[str, str], set[int]] = defaultdict(set)
    combo_correct_total: dict[tuple[str, str], set[int]] = defaultdict(set)
    combo_incorrect_total: dict[tuple[str, str], set[int]] = defaultdict(set)
    all_combos: set[tuple[str, str]] = set()

    for idx, pe_features in samples.items():
        label = labels[idx]
        for h in pe_features.heads:
            combo = (h.mode, h.segment)
            all_combos.add(combo)
            if label == "correct":
                combo_correct_total[combo].add(idx)
                if h.eligible:
                    combo_correct_eligible[combo].add(idx)
                if h.eligible and h.pe is not None:
                    combo_correct_pe_present[combo].add(idx)
            else:
                combo_incorrect_total[combo].add(idx)
                if h.eligible:
                    combo_incorrect_eligible[combo].add(idx)
                if h.eligible and h.pe is not None:
                    combo_incorrect_pe_present[combo].add(idx)

    # Build cells in deterministic order
    cells: list[EligibilityCell] = []
    for mode, segment in sorted(all_combos):
        cells.append(
            EligibilityCell(
                mode=mode,
                segment=segment,
                n_correct_eligible=len(combo_correct_eligible[(mode, segment)]),
                n_incorrect_eligible=len(combo_incorrect_eligible[(mode, segment)]),
                n_correct_pe_present=len(combo_correct_pe_present[(mode, segment)]),
                n_incorrect_pe_present=len(combo_incorrect_pe_present[(mode, segment)]),
                n_correct_total=len(combo_correct_total[(mode, segment)]),
                n_incorrect_total=len(combo_incorrect_total[(mode, segment)]),
            )
        )

    return EligibilityTable(cells=cells, n_correct=n_correct, n_incorrect=n_incorrect)
