"""Tests for eligibility accounting.

Proves: correct counting, asymmetric censoring detection,
fail-closed on bad inputs, no statistics computed.
"""

from __future__ import annotations

import pytest

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.types import EligibilityTable
from navi_sad.signal.pe_features import HeadPEResult, PEConfig, SamplePEFeatures


def _make_head(
    layer: int,
    head: int,
    mode: str,
    segment: str,
    eligible: bool = True,
    pe: float | None = 0.8,
) -> HeadPEResult:
    """Build a single HeadPEResult for testing."""
    return HeadPEResult(
        layer_idx=layer,
        head_idx=head,
        mode=mode,
        segment=segment,
        sequence_length=20 if eligible else 5,
        eligible=eligible,
        pe=pe,
        tie_rate=0.0,
        n_strict_patterns=10,
    )


def _make_sample(
    dataset_index: int,
    heads: list[HeadPEResult],
) -> SamplePEFeatures:
    return SamplePEFeatures(
        dataset_index=dataset_index,
        config=PEConfig(),
        heads=heads,
    )


class TestBuildEligibilityTable:
    def test_symmetric_eligibility(self) -> None:
        """All samples eligible in all combos."""
        heads = [_make_head(0, 0, "raw", "full")]
        samples = {
            10: _make_sample(10, heads),
            20: _make_sample(20, heads),
            30: _make_sample(30, heads),
        }
        labels = {10: "correct", 20: "correct", 30: "incorrect"}
        table = build_eligibility_table(samples, labels)
        assert isinstance(table, EligibilityTable)
        assert table.n_correct == 2
        assert table.n_incorrect == 1
        cell = table.cells[0]
        assert cell.mode == "raw"
        assert cell.segment == "full"
        assert cell.n_correct_eligible == 2
        assert cell.n_incorrect_eligible == 1

    def test_asymmetric_censoring(self) -> None:
        """Incorrect sample ineligible in a segment — censoring detected."""
        correct_heads = [
            _make_head(0, 0, "raw", "full", eligible=True, pe=0.8),
            _make_head(0, 0, "raw", "early", eligible=True, pe=0.7),
        ]
        incorrect_heads = [
            _make_head(0, 0, "raw", "full", eligible=True, pe=0.6),
            _make_head(0, 0, "raw", "early", eligible=False, pe=None),
        ]
        samples = {
            1: _make_sample(1, correct_heads),
            2: _make_sample(2, incorrect_heads),
        }
        labels = {1: "correct", 2: "incorrect"}
        table = build_eligibility_table(samples, labels)
        full_cell = next(c for c in table.cells if c.segment == "full")
        early_cell = next(c for c in table.cells if c.segment == "early")
        assert full_cell.n_incorrect_eligible == 1
        assert early_cell.n_incorrect_eligible == 0

    def test_pe_none_eligible_but_all_tied(self) -> None:
        """Eligible=True but pe=None (all ties). Eligible but not pe-present."""
        heads = [_make_head(0, 0, "raw", "full", eligible=True, pe=None)]
        samples = {1: _make_sample(1, heads)}
        labels = {1: "correct"}
        table = build_eligibility_table(samples, labels)
        cell = table.cells[0]
        assert cell.n_correct_eligible == 1  # passed length threshold
        assert cell.n_correct_pe_present == 0  # pe=None, did not contribute to d
        assert cell.n_correct_total == 1

    def test_unknown_label_raises(self) -> None:
        """Labels must be 'correct' or 'incorrect'. Anything else is rejected."""
        heads = [_make_head(0, 0, "raw", "full")]
        samples = {1: _make_sample(1, heads)}
        labels = {1: "ambiguous"}
        with pytest.raises(ValueError, match="unknown label"):
            build_eligibility_table(samples, labels)

    def test_missing_sample_label_raises(self) -> None:
        """Every sample must have a label."""
        heads = [_make_head(0, 0, "raw", "full")]
        samples = {1: _make_sample(1, heads)}
        labels: dict[int, str] = {}
        with pytest.raises(KeyError):
            build_eligibility_table(samples, labels)

    def test_empty_samples(self) -> None:
        """Empty input produces empty table."""
        table = build_eligibility_table({}, {})
        assert table.cells == []
        assert table.n_correct == 0
        assert table.n_incorrect == 0

    def test_multiple_heads_same_combo(self) -> None:
        """Multiple heads in same (mode, segment) — eligibility is per-sample.

        If ANY head in a combo is eligible for a sample, that sample counts as
        eligible for that combo.
        """
        heads = [
            _make_head(0, 0, "raw", "full", eligible=True, pe=0.8),
            _make_head(0, 1, "raw", "full", eligible=True, pe=0.7),
            _make_head(1, 0, "raw", "full", eligible=False, pe=None),
        ]
        samples = {1: _make_sample(1, heads)}
        labels = {1: "correct"}
        table = build_eligibility_table(samples, labels)
        cell = table.cells[0]
        # Sample has at least one eligible head in (raw, full)
        assert cell.n_correct_eligible == 1
