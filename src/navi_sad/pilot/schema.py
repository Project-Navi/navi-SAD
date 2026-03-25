"""Typed write-side schema for Gate 3 pilot artifacts.

Single source of truth for samples.json and review.json field
definitions on the write path. Uses frozen dataclasses with
__post_init__ validation for constrained values (enums). The review
record is derived from the sample record so field drift between
write-path artifacts is structurally impossible.

The read/analysis path still operates on raw dicts loaded from JSON.
Full end-to-end typed enforcement (including analysis) is deferred.

Design rule: dataclasses for repo-owned persisted artifacts.
Pydantic reserved for untrusted LLM boundary I/O (future judge).
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class Label(enum.Enum):
    """Human or scorer label for a generation."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


class DisagreementCategory(enum.Enum):
    """Reason for scorer/human label disagreement."""

    HEDGING = "hedging"
    CONTRADICTION = "contradiction"
    PARTIAL_MATCH = "partial-match"
    OFF_TOPIC = "off-topic"
    FORMAT_ISSUE = "format-issue"
    SCORER_TOO_STRICT = "scorer-too-strict"
    SCORER_TOO_LOOSE = "scorer-too-loose"


class StopReason(enum.Enum):
    """Why generation terminated."""

    EOS = "eos"
    MAX_LENGTH = "max_length"


class SpanStopReason(enum.Enum):
    """Why leading-span extraction stopped."""

    NEWLINE = "newline"
    PERIOD_SPACE = "period_space"
    PERIOD_NEWLINE = "period_newline"
    EOS = "eos"


def _validate_enum_field(value: str, enum_type: type[enum.Enum], field_name: str) -> None:
    """Validate that a string value is a valid member of an enum."""
    valid = {e.value for e in enum_type}
    if value not in valid:
        raise ValueError(f"Invalid {field_name}: {value!r}. Must be one of: {sorted(valid)}")


@dataclass(frozen=True)
class PilotSampleRecord:
    """Per-sample entry in samples.json (frozen after construction)."""

    # Identity
    dataset_index: int
    question: str
    best_answer: str
    correct_answers: tuple[str, ...]
    incorrect_answers: tuple[str, ...]

    # Generation
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    generated_token_count: int
    generation_text: str
    stop_reason: str

    # Measurement
    per_step: tuple[dict[str, Any], ...]
    full_gen_mean_delta: tuple[tuple[float, ...], ...] | None
    leading_span_mean_delta: tuple[tuple[float, ...], ...] | None
    leading_span_token_count: int
    leading_span_fallback: bool

    # Scoring
    scorer_label: str
    scorer_leading_span: str
    scorer_leading_span_stop_reason: str
    scorer_matched_correct: tuple[str, ...]
    scorer_matched_incorrect: tuple[str, ...]

    # Instrument health
    sample_error: str | None = None

    def __post_init__(self) -> None:
        """Validate constrained fields at construction time."""
        _validate_enum_field(self.stop_reason, StopReason, "stop_reason")
        _validate_enum_field(self.scorer_label, Label, "scorer_label")
        _validate_enum_field(
            self.scorer_leading_span_stop_reason,
            SpanStopReason,
            "scorer_leading_span_stop_reason",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict (lists, not tuples)."""
        d = asdict(self)
        # asdict converts tuples to lists, which is correct for JSON.
        return d


@dataclass
class PilotReviewRecord:
    """Per-sample entry in review.json (human-editable).

    All fields except the human-editable ones are copied from
    PilotSampleRecord and treated as read-only. Not frozen because
    human_label/disagreement fields are set during manual review.
    """

    # Copied from sample (read-only)
    dataset_index: int
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]
    rendered_prompt: str
    generation_text: str
    generated_token_count: int
    scorer_label: str
    scorer_leading_span: str
    scorer_leading_span_stop_reason: str
    scorer_matched_correct: list[str]
    scorer_matched_incorrect: list[str]

    # Human-editable
    human_label: str = ""
    disagreement_category: str = ""
    disagreement_note: str = ""

    def __post_init__(self) -> None:
        """Validate constrained fields at construction time."""
        _validate_enum_field(self.scorer_label, Label, "scorer_label")
        _validate_enum_field(
            self.scorer_leading_span_stop_reason,
            SpanStopReason,
            "scorer_leading_span_stop_reason",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)


@dataclass
class PilotMetadata:
    """Top-level metadata block in samples.json."""

    seed: int
    selected_indices: list[int]
    burned_indices: list[int]
    dataset_name: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    datasets_version: str
    dataset_fingerprint: str | None
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_hash: str
    transformers_version: str
    navi_sad_version: str
    decode_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)


@dataclass
class PilotSamplesArtifact:
    """Top-level structure of samples.json."""

    metadata: PilotMetadata
    samples: list[PilotSampleRecord]

    def write(self, path: Path) -> None:
        """Write to JSON file."""
        d = {"metadata": self.metadata.to_dict(), "samples": [s.to_dict() for s in self.samples]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        """Read raw dict from JSON file (for analysis, not reconstruction)."""
        with open(path, encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result


# --- Derived constants ---

# Fields that the reviewer may edit. Everything else in
# PilotReviewRecord is read-only and validated against samples.json.
HUMAN_EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "human_label",
        "disagreement_category",
        "disagreement_note",
    }
)

# Derived: read-only fields are all review fields minus the editable
# ones. dataset_index is the join key and validated separately.
REVIEW_READONLY_FIELDS: tuple[str, ...] = tuple(
    k
    for k in PilotReviewRecord.__dataclass_fields__
    if k not in HUMAN_EDITABLE_FIELDS and k != "dataset_index"
)

# Valid label values (derived from enum)
VALID_LABELS: frozenset[str] = frozenset(label.value for label in Label)

# Valid disagreement categories (derived from enum)
VALID_DISAGREEMENT_CATEGORIES: frozenset[str] = frozenset(cat.value for cat in DisagreementCategory)


def make_review_from_sample(sample: PilotSampleRecord) -> PilotReviewRecord:
    """Derive a review record from a sample record.

    Copies all shared fields. Sets human-editable fields to empty
    strings. This is the only way review records should be created --
    it guarantees field consistency with the sample schema.
    """
    return PilotReviewRecord(
        dataset_index=sample.dataset_index,
        question=sample.question,
        best_answer=sample.best_answer,
        correct_answers=list(sample.correct_answers),
        incorrect_answers=list(sample.incorrect_answers),
        rendered_prompt=sample.rendered_prompt,
        generation_text=sample.generation_text,
        generated_token_count=sample.generated_token_count,
        scorer_label=sample.scorer_label,
        scorer_leading_span=sample.scorer_leading_span,
        scorer_leading_span_stop_reason=sample.scorer_leading_span_stop_reason,
        scorer_matched_correct=list(sample.scorer_matched_correct),
        scorer_matched_incorrect=list(sample.scorer_matched_incorrect),
    )
