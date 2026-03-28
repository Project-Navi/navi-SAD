"""Tests for analysis artifact loader.

Proves: integrity validation, reject-not-drop semantics,
empty-after-filter rejection, correct filtering.
"""

from __future__ import annotations

import json

import pytest

from navi_sad.analysis.loader import AnalysisInput, load_and_validate
from navi_sad.core.types import StepRecord


def _write_artifacts(
    tmp_path: object,
    samples: list[dict],
    reviews: list[dict],
) -> object:
    """Write samples.json and review.json to tmp_path."""
    from pathlib import Path

    d = Path(str(tmp_path))
    with open(d / "samples.json", "w") as f:
        json.dump({"samples": samples}, f)
    with open(d / "review.json", "w") as f:
        json.dump(reviews, f)
    return d


# Read-only fields that must be consistent between sample and review.
_SHARED_FIELDS: dict[str, object] = {
    "question": "What is the capital?",
    "best_answer": "Paris",
    "correct_answers": ["Paris"],
    "incorrect_answers": ["London"],
    "rendered_prompt": "Q: What is the capital?",
    "generation_text": "Paris is the capital.",
    "generated_token_count": 100,
    "scorer_label": "correct",
    "scorer_leading_span": "Paris",
    "scorer_leading_span_stop_reason": "eos",
    "scorer_matched_correct": ["Paris"],
    "scorer_matched_incorrect": [],
}


def _make_sample(
    idx: int,
    *,
    generated_token_count: int = 100,
    sample_error: str | None = None,
) -> dict:
    """Build a sample dict with all fields needed by validate_review_integrity."""
    s: dict = {
        "dataset_index": idx,
        **_SHARED_FIELDS,
        "generated_token_count": generated_token_count,
        "per_step": [],
    }
    if sample_error is not None:
        s["sample_error"] = sample_error
    return s


def _make_review(
    idx: int,
    *,
    human_label: str = "correct",
    generated_token_count: int = 100,
) -> dict:
    """Build a review dict that passes validate_review_integrity.

    Read-only fields are copied from _SHARED_FIELDS to match sample fixtures.
    """
    r: dict = {
        "dataset_index": idx,
        **_SHARED_FIELDS,
        "generated_token_count": generated_token_count,
        "human_label": human_label,
    }
    # If human_label differs from scorer_label, add disagreement fields
    if human_label != _SHARED_FIELDS["scorer_label"]:
        r["disagreement_category"] = "hedging"
        r["disagreement_note"] = "test fixture"
    return r


class TestLoadAndValidate:
    def test_basic_load(self, tmp_path: object) -> None:
        samples = [_make_sample(1), _make_sample(2)]
        reviews = [_make_review(1, human_label="correct"), _make_review(2, human_label="incorrect")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        assert isinstance(result, AnalysisInput)
        assert result.n_correct == 1
        assert result.n_incorrect == 1
        assert set(result.labels.keys()) == {1, 2}

    def test_missing_samples_file(self, tmp_path: object) -> None:
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            load_and_validate(Path(str(tmp_path)))

    def test_duplicate_review_index_raises(self, tmp_path: object) -> None:
        """Duplicate index in review must raise, not silently pick one."""
        samples = [_make_sample(1)]
        reviews = [_make_review(1), _make_review(1)]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_and_validate(Path(str(d)))

    def test_missing_review_for_sample_raises(self, tmp_path: object) -> None:
        """Sample without a review must raise, not be silently dropped."""
        samples = [_make_sample(1), _make_sample(2)]
        reviews = [_make_review(1)]  # Missing review for idx=2
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        with pytest.raises(ValueError):
            load_and_validate(Path(str(d)))

    def test_excludes_ambiguous_labels(self, tmp_path: object) -> None:
        samples = [_make_sample(1), _make_sample(2), _make_sample(3)]
        reviews = [
            _make_review(1, human_label="correct"),
            _make_review(2, human_label="incorrect"),
            _make_review(3, human_label="ambiguous"),
        ]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        assert 3 not in result.labels
        assert result.n_correct == 1
        assert result.n_incorrect == 1

    def test_excludes_sample_errors(self, tmp_path: object) -> None:
        samples = [_make_sample(1), _make_sample(2, sample_error="generation failed")]
        reviews = [
            _make_review(1, human_label="correct"),
            _make_review(2, human_label="correct"),
        ]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        assert 2 not in result.labels

    def test_empty_after_filter_raises(self, tmp_path: object) -> None:
        """All samples ambiguous -> no analyzable data -> raise."""
        samples = [_make_sample(1)]
        reviews = [_make_review(1, human_label="ambiguous")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        with pytest.raises(ValueError, match="No analyzable samples"):
            load_and_validate(Path(str(d)))

    def test_malformed_per_step_raises(self, tmp_path: object) -> None:
        """Per-step records with missing required keys must be rejected at boundary."""
        s = _make_sample(1)
        s["per_step"] = [{"step_idx": 0}]  # missing layer_idx and per_head_delta
        samples = [s]
        reviews = [_make_review(1, human_label="correct")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        with pytest.raises(ValueError, match="missing required keys"):
            load_and_validate(Path(str(d)))

    def test_string_step_idx_raises(self, tmp_path: object) -> None:
        """step_idx must be int, not str."""
        s = _make_sample(1)
        s["per_step"] = [{"step_idx": "0", "layer_idx": 0, "per_head_delta": [0.1]}]
        d = _write_artifacts(tmp_path, [s], [_make_review(1)])
        from pathlib import Path

        with pytest.raises(ValueError, match="step_idx must be int"):
            load_and_validate(Path(str(d)))

    def test_none_layer_idx_raises(self, tmp_path: object) -> None:
        """layer_idx must be int, not None."""
        s = _make_sample(1)
        s["per_step"] = [{"step_idx": 0, "layer_idx": None, "per_head_delta": [0.1]}]
        d = _write_artifacts(tmp_path, [s], [_make_review(1)])
        from pathlib import Path

        with pytest.raises(ValueError, match="layer_idx must be int"):
            load_and_validate(Path(str(d)))

    def test_string_delta_element_raises(self, tmp_path: object) -> None:
        """per_head_delta elements must be numeric, not str."""
        s = _make_sample(1)
        s["per_step"] = [{"step_idx": 0, "layer_idx": 0, "per_head_delta": ["0.1", 0.2]}]
        d = _write_artifacts(tmp_path, [s], [_make_review(1)])
        from pathlib import Path

        with pytest.raises(ValueError, match=r"per_head_delta.*must be numeric"):
            load_and_validate(Path(str(d)))

    def test_provenance_paths_recorded(self, tmp_path: object) -> None:
        samples = [_make_sample(1)]
        reviews = [_make_review(1, human_label="correct")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        assert "samples.json" in result.samples_path
        assert "review.json" in result.review_path

    def test_per_step_parsed_to_step_records(self, tmp_path: object) -> None:
        """Per-step data must be parsed to StepRecord, not left as raw dicts."""
        s = _make_sample(1)
        s["per_step"] = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1, 0.2]},
            {"step_idx": 0, "layer_idx": 1, "per_head_delta": [0.3, 0.4]},
        ]
        samples = [s]
        reviews = [_make_review(1, human_label="correct")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        records = result.per_step_data[1]
        assert len(records) == 2
        assert isinstance(records[0], StepRecord)
        assert records[0].step_idx == 0
        assert records[0].layer_idx == 0
        assert records[0].per_head_delta == [0.1, 0.2]

    def test_step_record_extra_keys_ignored(self, tmp_path: object) -> None:
        """Extra keys in per-step dicts are dropped during StepRecord parsing."""
        s = _make_sample(1)
        s["per_step"] = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1], "extra": "ignored"},
        ]
        samples = [s]
        reviews = [_make_review(1, human_label="correct")]
        d = _write_artifacts(tmp_path, samples, reviews)
        from pathlib import Path

        result = load_and_validate(Path(str(d)))
        rec = result.per_step_data[1][0]
        assert isinstance(rec, StepRecord)
        assert not hasattr(rec, "extra")
