"""Tests for Gate 3 pilot analysis guards and integrity validation.

Covers: validate_review_integrity, compute_cohens_d,
compute_confusion_matrix.
"""

from __future__ import annotations

import pytest

from navi_sad.pilot.helpers import (
    compute_cohens_d,
    compute_confusion_matrix,
    validate_review_integrity,
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

    def test_all_readonly_fields_present(self) -> None:
        """Every field in _REVIEW_READONLY_FIELDS must exist in both fixtures."""
        from navi_sad.pilot.helpers import _REVIEW_READONLY_FIELDS

        sample = _make_sample(0)
        review = _make_review(0)
        for field in _REVIEW_READONLY_FIELDS:
            assert field in sample, f"_make_sample missing field: {field}"
            assert field in review, f"_make_review missing field: {field}"


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
