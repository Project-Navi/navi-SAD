"""Tests for Gate 3 pilot analysis guards and integrity validation.

Covers: validate_review_integrity, compute_cohens_d,
compute_confusion_matrix, invalid-sample filtering, sidecar shape.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from navi_sad.pilot.helpers import (
    compute_cohens_d,
    compute_confusion_matrix,
    validate_review_integrity,
)
from navi_sad.pilot.schema import (
    HUMAN_EDITABLE_FIELDS,
    REVIEW_READONLY_FIELDS,
    PilotReviewRecord,
    PilotSampleRecord,
    make_review_from_sample,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


def _make_sample(
    idx: int,
    scorer_label: str = "correct",
    generation_text: str = "Paris",
) -> dict:
    """Build a minimal samples.json entry."""
    return {
        "dataset_index": idx,
        "question": f"Question {idx}?",
        "best_answer": "Paris",
        "correct_answers": ["Paris"],
        "incorrect_answers": ["London"],
        "rendered_prompt": f"[INST] Question {idx}? [/INST]",
        "generation_text": generation_text,
        "generated_token_count": 5,
        "scorer_label": scorer_label,
        "scorer_leading_span": "Paris",
        "scorer_matched_correct": ["Paris"],
        "scorer_matched_incorrect": [],
        "scorer_leading_span_stop_reason": "eos",
    }


def _make_review(
    idx: int,
    human_label: str = "correct",
    scorer_label: str = "correct",
    disagreement_category: str = "",
    disagreement_note: str = "",
    generation_text: str = "Paris",
) -> dict:
    """Build a minimal review.json entry."""
    return {
        "dataset_index": idx,
        "question": f"Question {idx}?",
        "best_answer": "Paris",
        "correct_answers": ["Paris"],
        "incorrect_answers": ["London"],
        "rendered_prompt": f"[INST] Question {idx}? [/INST]",
        "generation_text": generation_text,
        "generated_token_count": 5,
        "scorer_label": scorer_label,
        "scorer_leading_span": "Paris",
        "scorer_matched_correct": ["Paris"],
        "scorer_matched_incorrect": [],
        "scorer_leading_span_stop_reason": "eos",
        "human_label": human_label,
        "disagreement_category": disagreement_category,
        "disagreement_note": disagreement_note,
    }


# -------------------------------------------------------------------
# validate_review_integrity
# -------------------------------------------------------------------


class TestReviewSchemaRoundTrip:
    """Verify generated review schema passes its own validator."""

    def test_generated_review_passes_integrity(self) -> None:
        """A review.json built from the same schema as the generation script
        must pass validate_review_integrity when human_label is filled."""
        # Simulate what run_generation() produces
        sample = _make_sample(42)
        review = _make_review(42, human_label="correct")
        # Should not raise — the generated schema must be self-consistent
        validate_review_integrity([review], [sample])

    def test_all_readonly_fields_present_in_fixtures(self) -> None:
        """Every field in REVIEW_READONLY_FIELDS must exist in both fixtures."""
        sample = _make_sample(0)
        review = _make_review(0)
        for field in REVIEW_READONLY_FIELDS:
            assert field in sample, f"_make_sample missing field: {field}"
            assert field in review, f"_make_review missing field: {field}"

    def test_readonly_fields_derived_from_schema(self) -> None:
        """REVIEW_READONLY_FIELDS must equal review fields minus editable ones."""
        all_review_fields = set(PilotReviewRecord.__dataclass_fields__.keys())
        expected_readonly = all_review_fields - HUMAN_EDITABLE_FIELDS - {"dataset_index"}
        assert set(REVIEW_READONLY_FIELDS) == expected_readonly

    def test_review_fields_are_subset_of_sample(self) -> None:
        """Every non-editable review field must exist in the sample schema."""
        sample_fields = set(PilotSampleRecord.__dataclass_fields__.keys())
        review_readonly = set(REVIEW_READONLY_FIELDS) | {"dataset_index"}
        missing = review_readonly - sample_fields
        assert not missing, f"Review fields not in sample schema: {missing}"

    def test_make_review_from_sample_copies_all_readonly(self) -> None:
        """make_review_from_sample must copy every readonly field."""
        sample = PilotSampleRecord(
            dataset_index=0,
            question="Question 0?",
            best_answer="Paris",
            correct_answers=["Paris"],
            incorrect_answers=["London"],
            rendered_prompt="[INST] Question 0? [/INST]",
            prompt_token_ids=[1, 2, 3],
            prompt_token_count=3,
            generated_token_ids=[4, 5],
            generated_token_count=2,
            generation_text="Paris",
            stop_reason="eos",
            per_step=[],
            full_gen_mean_delta=None,
            leading_span_mean_delta=None,
            leading_span_token_count=0,
            leading_span_fallback=False,
            scorer_label="correct",
            scorer_leading_span="Paris",
            scorer_leading_span_stop_reason="eos",
            scorer_matched_correct=["Paris"],
            scorer_matched_incorrect=[],
        )
        review = make_review_from_sample(sample)
        for fld in REVIEW_READONLY_FIELDS:
            assert hasattr(review, fld), f"make_review_from_sample missing: {fld}"
            assert getattr(review, fld) == getattr(sample, fld), (
                f"Field {fld} differs: review={getattr(review, fld)!r} "
                f"vs sample={getattr(sample, fld)!r}"
            )


class TestValidateReviewIntegrity:
    """Tests for review artifact integrity validation."""

    def test_valid_pair(self) -> None:
        samples = [_make_sample(0), _make_sample(1)]
        reviews = [_make_review(0), _make_review(1)]
        # Should not raise
        validate_review_integrity(reviews, samples)

    def test_missing_index_coverage(self) -> None:
        samples = [_make_sample(0), _make_sample(1)]
        reviews = [_make_review(0)]  # missing index 1
        with pytest.raises(ValueError, match="coverage"):
            validate_review_integrity(reviews, samples)

    def test_extra_index_in_review(self) -> None:
        samples = [_make_sample(0)]
        reviews = [_make_review(0), _make_review(1)]  # extra index 1
        with pytest.raises(ValueError, match="coverage"):
            validate_review_integrity(reviews, samples)

    def test_duplicate_index(self) -> None:
        samples = [_make_sample(0), _make_sample(1)]
        reviews = [_make_review(0), _make_review(0)]  # duplicate
        with pytest.raises(ValueError, match="duplicate"):
            validate_review_integrity(reviews, samples)

    def test_blank_human_label(self) -> None:
        samples = [_make_sample(0)]
        reviews = [_make_review(0, human_label="")]
        with pytest.raises(ValueError, match="human_label"):
            validate_review_integrity(reviews, samples)

    def test_invalid_human_label(self) -> None:
        samples = [_make_sample(0)]
        reviews = [_make_review(0, human_label="maybe")]
        with pytest.raises(ValueError, match="human_label"):
            validate_review_integrity(reviews, samples)

    def test_missing_disagreement_category(self) -> None:
        """When labels disagree, disagreement_category must be set."""
        samples = [_make_sample(0, scorer_label="correct")]
        reviews = [
            _make_review(
                0,
                human_label="incorrect",
                scorer_label="correct",
                disagreement_category="",
            )
        ]
        with pytest.raises(ValueError, match="disagreement_category"):
            validate_review_integrity(reviews, samples)

    def test_invalid_disagreement_category(self) -> None:
        samples = [_make_sample(0, scorer_label="correct")]
        reviews = [
            _make_review(
                0,
                human_label="incorrect",
                scorer_label="correct",
                disagreement_category="bad-value",
            )
        ]
        with pytest.raises(ValueError, match="disagreement_category"):
            validate_review_integrity(reviews, samples)

    def test_disagreement_category_ok_when_labels_agree(self) -> None:
        """No disagreement_category required when labels agree."""
        samples = [_make_sample(0, scorer_label="correct")]
        reviews = [_make_review(0, human_label="correct", scorer_label="correct")]
        # Should not raise
        validate_review_integrity(reviews, samples)

    def test_stale_disagreement_category_rejected(self) -> None:
        """Stale disagreement_category when labels agree must fail."""
        samples = [_make_sample(0, scorer_label="correct")]
        reviews = [
            _make_review(
                0,
                human_label="correct",
                scorer_label="correct",
                disagreement_category="hedging",
            )
        ]
        with pytest.raises(ValueError, match="disagreement_category"):
            validate_review_integrity(reviews, samples)

    def test_stale_disagreement_note_rejected(self) -> None:
        """Stale disagreement_note when labels agree must fail."""
        samples = [_make_sample(0, scorer_label="correct")]
        reviews = [
            _make_review(
                0,
                human_label="correct",
                scorer_label="correct",
                disagreement_note="leftover note",
            )
        ]
        with pytest.raises(ValueError, match="disagreement_note"):
            validate_review_integrity(reviews, samples)

    def test_invalid_scorer_label_rejected(self) -> None:
        """Invalid scorer_label in the artifact must fail."""
        samples = [_make_sample(0, scorer_label="maybe")]
        reviews = [_make_review(0, human_label="correct", scorer_label="maybe")]
        with pytest.raises(ValueError, match="scorer_label"):
            validate_review_integrity(reviews, samples)

    def test_readonly_field_drift(self) -> None:
        """Changing a read-only field should fail."""
        samples = [_make_sample(0, generation_text="Paris")]
        reviews = [_make_review(0, generation_text="London")]
        with pytest.raises(ValueError, match="drift"):
            validate_review_integrity(reviews, samples)

    def test_readonly_field_missing_from_review(self) -> None:
        """Deleting a read-only field from review should fail."""
        samples = [_make_sample(0)]
        reviews = [_make_review(0)]
        del reviews[0]["generation_text"]
        with pytest.raises(ValueError, match="missing from review"):
            validate_review_integrity(reviews, samples)


# -------------------------------------------------------------------
# compute_cohens_d
# -------------------------------------------------------------------


class TestComputeCohensD:
    """Tests for guarded Cohen's d computation."""

    def test_valid_groups(self) -> None:
        value, reason = compute_cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert value is not None
        assert reason is None
        # mean_a=2, mean_b=5, pooled_sd=1 -> d = (2-5)/1 = -3.0
        assert value == pytest.approx(-3.0)

    def test_group_a_too_small(self) -> None:
        value, reason = compute_cohens_d([1.0], [2.0, 3.0, 4.0])
        assert value is None
        assert reason is not None
        assert "group_a" in reason

    def test_group_b_too_small(self) -> None:
        value, reason = compute_cohens_d([1.0, 2.0, 3.0], [4.0])
        assert value is None
        assert reason is not None
        assert "group_b" in reason

    def test_zero_variance(self) -> None:
        value, reason = compute_cohens_d([5.0, 5.0], [5.0, 5.0])
        assert value is None
        assert reason is not None
        assert "variance" in reason

    def test_minimum_valid_size(self) -> None:
        """Two samples per group is the minimum."""
        value, reason = compute_cohens_d([1.0, 2.0], [3.0, 4.0])
        assert value is not None
        assert reason is None


# -------------------------------------------------------------------
# compute_confusion_matrix
# -------------------------------------------------------------------


class TestComputeConfusionMatrix:
    """Tests for 3x3 confusion matrix with guarded precision/recall."""

    def test_perfect_agreement(self) -> None:
        scorer = ["correct", "incorrect", "ambiguous"]
        human = ["correct", "incorrect", "ambiguous"]
        result = compute_confusion_matrix(scorer, human)
        assert result["overall_agreement"] == pytest.approx(1.0)
        assert result["matrix"]["correct"]["correct"] == 1
        assert result["matrix"]["incorrect"]["incorrect"] == 1
        assert result["matrix"]["ambiguous"]["ambiguous"] == 1

    def test_all_cells_populated(self) -> None:
        scorer = ["correct", "incorrect", "ambiguous"] * 3
        human = [
            "correct",
            "correct",
            "correct",
            "incorrect",
            "incorrect",
            "incorrect",
            "ambiguous",
            "ambiguous",
            "ambiguous",
        ]
        result = compute_confusion_matrix(scorer, human)
        # Each scorer label predicted 3 times
        # Each human label assigned 3 times
        assert result["overall_agreement"] == pytest.approx(1.0 / 3.0)

    def test_zero_support_precision(self) -> None:
        """Precision is null when no samples predicted as that class."""
        scorer = ["correct", "correct"]
        human = ["correct", "incorrect"]
        result = compute_confusion_matrix(scorer, human)
        # "incorrect" never predicted -> precision is null
        prec_value, prec_reason = result["per_class"]["incorrect"]["precision"]
        assert prec_value is None
        assert prec_reason is not None

    def test_zero_support_recall(self) -> None:
        """Recall is null when no human labels for that class."""
        scorer = ["correct", "incorrect"]
        human = ["correct", "correct"]
        result = compute_confusion_matrix(scorer, human)
        # "incorrect" never in human labels -> recall is null
        rec_value, rec_reason = result["per_class"]["incorrect"]["recall"]
        assert rec_value is None
        assert rec_reason is not None

    def test_agreement_rate(self) -> None:
        scorer = ["correct", "correct", "incorrect"]
        human = ["correct", "incorrect", "incorrect"]
        result = compute_confusion_matrix(scorer, human)
        assert result["overall_agreement"] == pytest.approx(2.0 / 3.0)

    def test_empty_inputs(self) -> None:
        result = compute_confusion_matrix([], [])
        assert result["overall_agreement"] == pytest.approx(0.0)


# -------------------------------------------------------------------
# Invalid-sample analysis behavior
# -------------------------------------------------------------------


class TestInvalidSampleAnalysis:
    """Verify --analyze handles invalid samples consistently."""

    def _make_analysis_artifacts(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create samples.json + review.json with one valid and one invalid sample."""
        s_valid = _make_sample(0, scorer_label="correct")
        s_valid["sample_error"] = None
        s_valid["full_gen_mean_delta"] = [[0.5, 0.6]]
        s_valid["per_step"] = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.5, 0.6]},
        ]
        s_valid["stop_reason"] = "eos"
        s_valid["leading_span_fallback"] = False
        s_valid["scorer_leading_span_stop_reason"] = "eos"

        s_invalid = _make_sample(1, scorer_label="incorrect")
        s_invalid["sample_error"] = "full_gen_mean_delta: Layer 0 has no records"
        s_invalid["full_gen_mean_delta"] = None
        s_invalid["per_step"] = []
        s_invalid["stop_reason"] = "eos"
        s_invalid["leading_span_fallback"] = False
        s_invalid["scorer_leading_span_stop_reason"] = "eos"

        samples_artifact = {
            "metadata": {"seed": 42, "selected_indices": [0, 1]},
            "samples": [s_valid, s_invalid],
        }
        samples_path = tmp_path / "samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples_artifact, f)

        r_valid = _make_review(0, human_label="correct", scorer_label="correct")
        r_valid["scorer_leading_span_stop_reason"] = "eos"

        r_invalid = _make_review(1, human_label="incorrect", scorer_label="incorrect")
        r_invalid["scorer_leading_span_stop_reason"] = "eos"

        review_path = tmp_path / "review.json"
        with open(review_path, "w") as f:
            json.dump([r_valid, r_invalid], f)

        return samples_path, review_path

    def test_analyze_reports_invalid_count(self, tmp_path: Path) -> None:
        """--analyze should warn about invalid samples."""
        _, review_path = self._make_analysis_artifacts(tmp_path)
        result = subprocess.run(
            [sys.executable, "scripts/pilot_gate3.py", "--analyze", str(review_path)],
            capture_output=True,
            text=True,
        )
        assert "1 invalid sample" in result.stdout
        assert "excluded from ALL analysis" in result.stdout

    def test_analyze_excludes_invalid_from_class_balance(self, tmp_path: Path) -> None:
        """Invalid samples should not appear in class balance counts."""
        _, review_path = self._make_analysis_artifacts(tmp_path)
        result = subprocess.run(
            [sys.executable, "scripts/pilot_gate3.py", "--analyze", str(review_path)],
            capture_output=True,
            text=True,
        )
        # Only 1 valid sample (correct), so incorrect count should be 0
        lines = result.stdout.splitlines()
        incorrect_lines = [ln for ln in lines if ln.strip().startswith("incorrect:")]
        assert any("0" in ln for ln in incorrect_lines)


# -------------------------------------------------------------------
# Sidecar output shape
# -------------------------------------------------------------------


class TestSidecarShape:
    """Verify cohens_d.json sidecar has required fields."""

    def test_sidecar_has_exploratory_note(self, tmp_path: Path) -> None:
        """cohens_d.json must contain the exploratory disclaimer."""
        # Build minimal artifacts with enough structure for --analyze
        s0 = _make_sample(0, scorer_label="correct")
        s0["sample_error"] = None
        s0["full_gen_mean_delta"] = [[0.5, 0.6]]
        s0["leading_span_mean_delta"] = [[0.4, 0.5]]
        s0["leading_span_fallback"] = False
        s0["per_step"] = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.5, 0.6]},
        ]
        s0["stop_reason"] = "eos"
        s0["scorer_leading_span_stop_reason"] = "eos"

        s1 = _make_sample(1, scorer_label="incorrect")
        s1["sample_error"] = None
        s1["full_gen_mean_delta"] = [[0.7, 0.8]]
        s1["leading_span_mean_delta"] = [[0.6, 0.7]]
        s1["leading_span_fallback"] = False
        s1["per_step"] = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.7, 0.8]},
        ]
        s1["stop_reason"] = "eos"
        s1["scorer_leading_span_stop_reason"] = "eos"

        samples_artifact = {
            "metadata": {"seed": 42, "selected_indices": [0, 1]},
            "samples": [s0, s1],
        }
        samples_path = tmp_path / "samples.json"
        with open(samples_path, "w") as f:
            json.dump(samples_artifact, f)

        r0 = _make_review(0, human_label="correct", scorer_label="correct")
        r0["scorer_leading_span_stop_reason"] = "eos"
        r1 = _make_review(
            1,
            human_label="incorrect",
            scorer_label="incorrect",
            disagreement_category="",
        )
        r1["scorer_leading_span_stop_reason"] = "eos"

        review_path = tmp_path / "review.json"
        with open(review_path, "w") as f:
            json.dump([r0, r1], f)

        subprocess.run(
            [sys.executable, "scripts/pilot_gate3.py", "--analyze", str(review_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        d_path = tmp_path / "cohens_d.json"
        assert d_path.exists()
        with open(d_path) as f:
            sidecar = json.load(f)

        assert "exploratory_note" in sidecar
        assert "EXPLORATORY" in sidecar["exploratory_note"]
        assert "full_gen_cohens_d" in sidecar
        assert "leading_span_cohens_d" in sidecar
        assert "n_correct_full" in sidecar
        assert "n_incorrect_full" in sidecar
