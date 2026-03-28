"""Tests for PE data preparation.

Proves: series data is D-independent, PE bundles are D-specific,
deterministic output, correct type wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

from navi_sad.analysis.prep import (
    PEBundle,
    SeriesData,
    compute_pe_bundle,
    prepare_series_data,
    prepare_series_data_from_subset,
)
from navi_sad.core.types import StepRecord
from navi_sad.signal.pe_features import PEConfig

# Shared fixture constants matching test_loader.py
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


def _make_sample(idx: int, num_steps: int = 20, num_layers: int = 2, num_heads: int = 2) -> dict:
    """Build a sample with realistic per-step data."""
    per_step = []
    for step in range(num_steps):
        for layer in range(num_layers):
            per_step.append(
                {
                    "step_idx": step,
                    "layer_idx": layer,
                    "per_head_delta": [
                        0.1 + 0.01 * step + 0.001 * layer * h for h in range(num_heads)
                    ],
                }
            )
    return {
        "dataset_index": idx,
        **_SHARED_FIELDS,
        "generated_token_count": num_steps,
        "per_step": per_step,
    }


def _make_review(
    idx: int, *, human_label: str = "correct", generated_token_count: int = 100
) -> dict:
    sl = _SHARED_FIELDS["scorer_label"]
    r: dict = {
        "dataset_index": idx,
        **_SHARED_FIELDS,
        "generated_token_count": generated_token_count,
        "human_label": human_label,
    }
    if human_label != sl:
        r["disagreement_category"] = "hedging"
        r["disagreement_note"] = "test fixture"
    return r


NUM_STEPS = 20


def _write_fixtures(tmp_path: Path, n_correct: int = 3, n_incorrect: int = 3) -> Path:
    """Write sample/review fixtures and return the directory."""
    samples = []
    reviews = []
    for i in range(n_correct):
        samples.append(_make_sample(i, num_steps=NUM_STEPS, num_layers=2, num_heads=2))
        reviews.append(_make_review(i, human_label="correct", generated_token_count=NUM_STEPS))
    for i in range(n_correct, n_correct + n_incorrect):
        samples.append(_make_sample(i, num_steps=NUM_STEPS, num_layers=2, num_heads=2))
        reviews.append(_make_review(i, human_label="incorrect", generated_token_count=NUM_STEPS))

    with open(tmp_path / "samples.json", "w") as f:
        json.dump({"samples": samples}, f)
    with open(tmp_path / "review.json", "w") as f:
        json.dump(reviews, f)
    return tmp_path


class TestPrepareSeriesData:
    def test_returns_series_data(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        assert isinstance(sd, SeriesData)
        assert sd.num_layers == 2
        assert sd.num_heads == 2

    def test_input_has_typed_step_records(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        for _idx, records in sd.input.per_step_data.items():
            assert all(isinstance(r, StepRecord) for r in records)

    def test_per_step_dicts_are_plain_dicts(self, tmp_path: Path) -> None:
        """per_step_dicts field holds dict-form data for PE API."""
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        for _idx, dicts in sd.per_step_dicts.items():
            assert all(isinstance(r, dict) for r in dicts)

    def test_baseline_computed(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        assert len(sd.baseline) > 0
        assert (0, 0) in sd.baseline

    def test_head_series_extracted(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        assert len(sd.head_series) == 6  # 3 correct + 3 incorrect


class TestComputePEBundle:
    def test_returns_pe_bundle(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        bundle = compute_pe_bundle(sd)
        assert isinstance(bundle, PEBundle)
        assert bundle.pe_config.D == 3

    def test_custom_d(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path, n_correct=5, n_incorrect=5)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        bundle = compute_pe_bundle(sd, pe_config=PEConfig(D=3))
        assert bundle.pe_config.D == 3

    def test_has_pe_samples_and_lookup(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        bundle = compute_pe_bundle(sd)
        assert len(bundle.pe_samples) == 6
        assert len(bundle.lookup) > 0

    def test_has_eligibility(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        bundle = compute_pe_bundle(sd)
        assert bundle.eligibility.n_correct == 3
        assert bundle.eligibility.n_incorrect == 3

    def test_deterministic(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        sd = prepare_series_data(d, num_layers=2, num_heads=2)
        b1 = compute_pe_bundle(sd)
        b2 = compute_pe_bundle(sd)
        # Same PE values
        for idx in b1.pe_samples:
            for h1, h2 in zip(b1.pe_samples[idx].heads, b2.pe_samples[idx].heads, strict=True):
                assert h1.pe == h2.pe


class TestPrepareSeriesDataFromSubset:
    def test_filters_to_indices(self, tmp_path: Path) -> None:
        """Only samples in the subset appear in the result."""
        d = _write_fixtures(tmp_path, n_correct=3, n_incorrect=3)
        full_sd = prepare_series_data(d, num_layers=2, num_heads=2)
        subset = prepare_series_data_from_subset(
            full_sd.input,
            indices={0, 1, 3},
            baseline=full_sd.baseline,
            num_layers=2,
            num_heads=2,
        )
        assert set(subset.head_series.keys()) == {0, 1, 3}
        assert set(subset.per_step_dicts.keys()) == {0, 1, 3}
        assert set(subset.input.labels.keys()) == {0, 1, 3}

    def test_uses_provided_baseline(self, tmp_path: Path) -> None:
        """Baseline is NOT recomputed — uses the one passed in."""
        d = _write_fixtures(tmp_path, n_correct=3, n_incorrect=3)
        full_sd = prepare_series_data(d, num_layers=2, num_heads=2)
        subset = prepare_series_data_from_subset(
            full_sd.input,
            indices={0, 1},
            baseline=full_sd.baseline,
            num_layers=2,
            num_heads=2,
        )
        # Baseline should be the same object (not recomputed)
        assert subset.baseline is full_sd.baseline

    def test_empty_indices_raises(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        full_sd = prepare_series_data(d, num_layers=2, num_heads=2)
        import pytest

        with pytest.raises(ValueError, match=r"[Ee]mpty"):
            prepare_series_data_from_subset(
                full_sd.input,
                indices=set(),
                baseline=full_sd.baseline,
                num_layers=2,
                num_heads=2,
            )

    def test_invalid_index_raises(self, tmp_path: Path) -> None:
        d = _write_fixtures(tmp_path)
        full_sd = prepare_series_data(d, num_layers=2, num_heads=2)
        import pytest

        with pytest.raises(ValueError, match="not in"):
            prepare_series_data_from_subset(
                full_sd.input,
                indices={999},
                baseline=full_sd.baseline,
                num_layers=2,
                num_heads=2,
            )

    def test_labels_filtered(self, tmp_path: Path) -> None:
        """Labels and token_counts reflect only the subset."""
        d = _write_fixtures(tmp_path, n_correct=3, n_incorrect=3)
        full_sd = prepare_series_data(d, num_layers=2, num_heads=2)
        subset = prepare_series_data_from_subset(
            full_sd.input,
            indices={0, 3},  # 1 correct + 1 incorrect
            baseline=full_sd.baseline,
            num_layers=2,
            num_heads=2,
        )
        assert subset.input.n_correct == 1
        assert subset.input.n_incorrect == 1
