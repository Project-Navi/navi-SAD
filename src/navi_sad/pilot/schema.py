"""Typed schema for Gate 3 pilot artifacts.

Single source of truth for samples.json and review.json field
definitions. Uses dataclasses for runtime enforcement and enums
for constrained values. The review record is derived from the
sample record so field drift is structurally impossible.

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


@dataclass
class PilotSampleRecord:
    """Per-sample entry in samples.json (immutable after generation)."""

    # Identity
    dataset_index: int
    question: str
    best_answer: str
    correct_answers: list[str]
    incorrect_answers: list[str]

    # Generation
    rendered_prompt: str
    prompt_token_ids: list[int]
    prompt_token_count: int
    generated_token_ids: list[int]
    generated_token_count: int
    generation_text: str
    stop_reason: str  # StopReason.value

    # Measurement
    per_step: list[dict[str, Any]]
    full_gen_mean_delta: list[list[float]] | None
    leading_span_mean_delta: list[list[float]] | None
    leading_span_token_count: int
    leading_span_fallback: bool

    # Scoring
    scorer_label: str  # Label.value
    scorer_leading_span: str
    scorer_leading_span_stop_reason: str  # SpanStopReason.value
    scorer_matched_correct: list[str]
    scorer_matched_incorrect: list[str]

    # Instrument health
    sample_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)


@dataclass
class PilotReviewRecord:
    """Per-sample entry in review.json (human-editable).

    All fields except the human-editable ones are copied from
    PilotSampleRecord and treated as read-only.
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
    scorer_label: str  # Label.value
    scorer_leading_span: str
    scorer_leading_span_stop_reason: str  # SpanStopReason.value
    scorer_matched_correct: list[str]
    scorer_matched_incorrect: list[str]

    # Human-editable
    human_label: str = ""
    disagreement_category: str = ""
    disagreement_note: str = ""

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
        correct_answers=sample.correct_answers,
        incorrect_answers=sample.incorrect_answers,
        rendered_prompt=sample.rendered_prompt,
        generation_text=sample.generation_text,
        generated_token_count=sample.generated_token_count,
        scorer_label=sample.scorer_label,
        scorer_leading_span=sample.scorer_leading_span,
        scorer_leading_span_stop_reason=sample.scorer_leading_span_stop_reason,
        scorer_matched_correct=sample.scorer_matched_correct,
        scorer_matched_incorrect=sample.scorer_matched_incorrect,
    )
