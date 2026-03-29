"""Load and validate pilot artifacts for PE recurrence analysis.

Boundary module: raw JSON -> validated structures. All filtering
decisions are explicit and tested. Rejects on integrity violations
instead of silently subsetting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from navi_sad.core.types import StepRecord


@dataclass(frozen=True)
class AnalysisInput:
    """Validated input for PE recurrence null analysis.

    All filtering decisions have already been applied and validated.
    """

    labels: dict[int, str]
    token_counts: dict[int, int]
    per_step_data: dict[int, list[StepRecord]]
    n_correct: int
    n_incorrect: int
    samples_path: str
    review_path: str


_PER_STEP_REQUIRED_KEYS = {"step_idx", "layer_idx", "per_head_delta"}


def _parse_per_step_records(
    raw_per_step: dict[int, list[dict[str, Any]]],
) -> dict[int, list[StepRecord]]:
    """Validate and parse per-step dicts into StepRecord objects.

    Terminates the raw JSON boundary: after this function, all per-step
    data is typed. Extra keys in the raw dicts are dropped.

    Raises:
        ValueError: If any record is missing required keys or has wrong types.
    """
    parsed: dict[int, list[StepRecord]] = {}
    for idx, records in raw_per_step.items():
        if not isinstance(records, list):
            raise ValueError(f"Sample {idx}: per_step must be a list, got {type(records).__name__}")
        sample_records: list[StepRecord] = []
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(
                    f"Sample {idx}, record {i}: per_step entry must be a dict, "
                    f"got {type(rec).__name__}"
                )
            missing = _PER_STEP_REQUIRED_KEYS - set(rec.keys())
            if missing:
                raise ValueError(
                    f"Sample {idx}, record {i}: missing required keys: {sorted(missing)}"
                )
            # Type validation: int fields must be int (not bool, not str)
            for int_field in ("step_idx", "layer_idx"):
                val = rec[int_field]
                if not isinstance(val, int) or isinstance(val, bool):
                    raise ValueError(
                        f"Sample {idx}, record {i}: {int_field} must be int, "
                        f"got {type(val).__name__}: {val!r}"
                    )
            if not isinstance(rec["per_head_delta"], list):
                raise ValueError(
                    f"Sample {idx}, record {i}: per_head_delta must be a list, "
                    f"got {type(rec['per_head_delta']).__name__}"
                )
            for j, delta in enumerate(rec["per_head_delta"]):
                if not isinstance(delta, (int, float)) or isinstance(delta, bool):
                    raise ValueError(
                        f"Sample {idx}, record {i}: per_head_delta[{j}] must be "
                        f"numeric, got {type(delta).__name__}: {delta!r}"
                    )
            # Parse to typed StepRecord — boundary terminates here
            sample_records.append(
                StepRecord(
                    step_idx=rec["step_idx"],
                    layer_idx=rec["layer_idx"],
                    per_head_delta=rec["per_head_delta"],
                )
            )
        parsed[idx] = sample_records
    return parsed


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
    raw_per_step = {s["dataset_index"]: s["per_step"] for s in included}

    # Parse per-step dicts to typed StepRecord objects at the boundary.
    # Validates shape + types and terminates the raw JSON boundary.
    per_step_data = _parse_per_step_records(raw_per_step)

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


_VALID_REVIEWER_LABELS = frozenset({"correct", "incorrect", "ambiguous"})


def load_reviewer_votes(labeling_dir: Path) -> dict[int, list[str]]:
    """Load per-reviewer vote data from labeling batch files.

    Reads batch_*_reviewer_*.json files. Each file is a list of dicts
    with 'dataset_index' and 'human_label'.

    Returns:
        {dataset_index: [reviewer_0_label, reviewer_1_label, ...]}.
        Reviewers are ordered by reviewer number from the filename.

    Raises:
        FileNotFoundError: If labeling_dir does not exist.
        ValueError: If no batch files found, reviewer counts are
            inconsistent, or labels contain invalid values.
    """
    if not labeling_dir.exists():
        raise FileNotFoundError(f"Labeling directory not found: {labeling_dir}")

    # Discover batch files
    import re

    pattern = re.compile(r"batch_(\d+)_reviewer_(\d+)\.json$")
    file_map: dict[tuple[int, int], Path] = {}
    for p in sorted(labeling_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            batch_idx = int(m.group(1))
            reviewer_idx = int(m.group(2))
            file_map[(batch_idx, reviewer_idx)] = p

    if not file_map:
        raise ValueError(f"No batch_*_reviewer_*.json files found in {labeling_dir}")

    # Determine reviewer set per batch
    batches: dict[int, set[int]] = {}
    for batch_idx, reviewer_idx in file_map:
        batches.setdefault(batch_idx, set()).add(reviewer_idx)

    reviewer_counts = {b: len(revs) for b, revs in batches.items()}
    unique_counts = set(reviewer_counts.values())
    if len(unique_counts) > 1:
        raise ValueError(
            f"Reviewer count varies across batches: {reviewer_counts}. "
            f"Expected consistent reviewer count."
        )

    n_reviewers = unique_counts.pop()

    # Load all votes
    votes: dict[int, list[str | None]] = {}  # idx -> [None]*n_reviewers initially
    for (_batch_idx, reviewer_idx), path in sorted(file_map.items()):
        with open(path, encoding="utf-8") as f:
            records: list[dict[str, Any]] = json.load(f)
        for rec in records:
            idx = rec["dataset_index"]
            label = rec["human_label"]
            if label not in _VALID_REVIEWER_LABELS:
                raise ValueError(
                    f"Invalid label {label!r} for dataset_index={idx} "
                    f"in {path.name}. Valid labels: {sorted(_VALID_REVIEWER_LABELS)}"
                )
            if idx not in votes:
                votes[idx] = [None] * n_reviewers  # type: ignore[list-item]
            votes[idx][reviewer_idx] = label

    # Validate: no missing reviewer slots
    result: dict[int, list[str]] = {}
    for idx in sorted(votes):
        slot = votes[idx]
        missing = [i for i, v in enumerate(slot) if v is None]
        if missing:
            raise ValueError(f"dataset_index={idx} missing votes from reviewers: {missing}")
        result[idx] = [v for v in slot if v is not None]

    return result


def step_records_to_dicts(records: list[StepRecord]) -> list[dict[str, Any]]:
    """Convert typed StepRecords back to dicts for PE API calls.

    The PE module's extract_head_sad_series accepts list[dict[str, Any]].
    This explicit conversion bridges the typed boundary with the existing API.
    """
    return [asdict(r) for r in records]
