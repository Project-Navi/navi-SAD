"""Load and validate pilot artifacts for PE recurrence analysis.

Boundary module: raw JSON -> validated structures. All filtering
decisions are explicit and tested. Rejects on integrity violations
instead of silently subsetting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisInput:
    """Validated input for PE recurrence null analysis.

    All filtering decisions have already been applied and validated.
    """

    labels: dict[int, str]
    token_counts: dict[int, int]
    per_step_data: dict[int, list[dict[str, Any]]]
    n_correct: int
    n_incorrect: int
    samples_path: str
    review_path: str


def load_and_validate(
    results_dir: Path,
) -> AnalysisInput:
    """Load pilot artifacts, validate integrity, filter to analyzable samples.

    Applies the following filters (in order):
    1. Reject if review/samples integrity fails (1:1 coverage, no duplicates,
       valid labels, readonly field consistency).
    2. Exclude samples with sample_error != None.
    3. Exclude samples with label not in {"correct", "incorrect"} (ambiguous).

    Args:
        results_dir: Directory containing samples.json and review.json.

    Returns:
        Validated AnalysisInput.

    Raises:
        FileNotFoundError: If samples.json or review.json is missing.
        ValueError: If review/samples integrity fails, or if no
            analyzable samples remain after filtering.
    """
    samples_path = results_dir / "samples.json"
    review_path = results_dir / "review.json"

    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Missing {review_path}")

    with open(samples_path, encoding="utf-8") as f:
        samples_artifact: dict[str, Any] = json.load(f)
    with open(review_path, encoding="utf-8") as f:
        review_data: list[dict[str, Any]] = json.load(f)

    samples_raw: list[dict[str, Any]] = samples_artifact["samples"]

    # Validate review/samples integrity before any filtering.
    # This catches duplicates, missing indices, label drift, and
    # readonly field mismatches. Raises ValueError on failure.
    from navi_sad.pilot.helpers import validate_review_integrity

    validate_review_integrity(review_data, samples_raw)

    # Build label lookup from validated review data
    labels_raw = {r["dataset_index"]: r["human_label"] for r in review_data}

    # Filter: correct/incorrect only, no sample errors
    included = [
        s
        for s in samples_raw
        if labels_raw.get(s["dataset_index"]) in ("correct", "incorrect")
        and s.get("sample_error") is None
    ]

    if not included:
        raise ValueError(
            f"No analyzable samples in {results_dir} after filtering. "
            f"Total samples: {len(samples_raw)}, "
            f"with labels: {len(labels_raw)}"
        )

    labels = {s["dataset_index"]: labels_raw[s["dataset_index"]] for s in included}
    token_counts = {s["dataset_index"]: s["generated_token_count"] for s in included}
    per_step_data = {s["dataset_index"]: s["per_step"] for s in included}

    n_correct = sum(1 for v in labels.values() if v == "correct")
    n_incorrect = sum(1 for v in labels.values() if v == "incorrect")

    return AnalysisInput(
        labels=labels,
        token_counts=token_counts,
        per_step_data=per_step_data,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        samples_path=str(samples_path),
        review_path=str(review_path),
    )
