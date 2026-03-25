"""Tests for SAD-specific temporal PE features.

Covers: extract_head_sad_series, sequence transforms,
compute_head_pe, compute_sample_pe_features,
compute_positional_baseline.
"""

from __future__ import annotations

import pytest

from navi_sad.signal.pe_features import (
    PEConfig,
    _detrend_by_baseline,
    _first_difference,
    _segment,
    compute_head_pe,
    compute_positional_baseline,
    compute_sample_pe_features,
    extract_head_sad_series,
)

# -------------------------------------------------------------------
# extract_head_sad_series
# -------------------------------------------------------------------


class TestExtractHeadSadSeries:
    """Tests for per-(layer, head) series extraction."""

    def test_basic_extraction(self) -> None:
        per_step = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1, 0.2]},
            {"step_idx": 0, "layer_idx": 1, "per_head_delta": [0.3, 0.4]},
            {"step_idx": 1, "layer_idx": 0, "per_head_delta": [0.5, 0.6]},
            {"step_idx": 1, "layer_idx": 1, "per_head_delta": [0.7, 0.8]},
        ]
        result = extract_head_sad_series(per_step, num_layers=2, num_heads=2)
        assert result[(0, 0)] == [0.1, 0.5]
        assert result[(0, 1)] == [0.2, 0.6]
        assert result[(1, 0)] == [0.3, 0.7]
        assert result[(1, 1)] == [0.4, 0.8]

    def test_empty_per_step(self) -> None:
        result = extract_head_sad_series([], num_layers=1, num_heads=1)
        assert result[(0, 0)] == []

    def test_step_ordering(self) -> None:
        """Steps should be ordered by step_idx regardless of input order."""
        per_step = [
            {"step_idx": 2, "layer_idx": 0, "per_head_delta": [0.9]},
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1]},
            {"step_idx": 1, "layer_idx": 0, "per_head_delta": [0.5]},
        ]
        result = extract_head_sad_series(per_step, num_layers=1, num_heads=1)
        assert result[(0, 0)] == [0.1, 0.5, 0.9]

    def test_duplicate_layer_step_raises(self) -> None:
        """Duplicate (layer_idx, step_idx) records must raise, not overwrite."""
        per_step = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1]},
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.9]},  # duplicate
        ]
        with pytest.raises(ValueError, match="duplicate"):
            extract_head_sad_series(per_step, num_layers=1, num_heads=1)

    def test_non_contiguous_steps_raises(self) -> None:
        """Missing step_idx within a layer must raise, not compress gaps."""
        per_step = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1]},
            # step_idx 1 missing
            {"step_idx": 2, "layer_idx": 0, "per_head_delta": [0.9]},
        ]
        with pytest.raises(ValueError, match="non-contiguous"):
            extract_head_sad_series(per_step, num_layers=1, num_heads=1)

    def test_out_of_range_layer_raises(self) -> None:
        """Layer index >= num_layers must raise, not be silently ignored."""
        per_step = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1]},
            {"step_idx": 0, "layer_idx": 5, "per_head_delta": [0.9]},  # out of range
        ]
        with pytest.raises(ValueError, match="layer_idx"):
            extract_head_sad_series(per_step, num_layers=2, num_heads=1)

    def test_wrong_head_width_raises(self) -> None:
        """per_head_delta length != num_heads must raise."""
        per_step = [
            {"step_idx": 0, "layer_idx": 0, "per_head_delta": [0.1, 0.2, 0.3]},
        ]
        with pytest.raises(ValueError, match="per_head_delta"):
            extract_head_sad_series(per_step, num_layers=1, num_heads=2)


# -------------------------------------------------------------------
# Sequence transforms
# -------------------------------------------------------------------


class TestFirstDifference:
    def test_basic(self) -> None:
        assert _first_difference([1.0, 3.0, 6.0, 10.0]) == [2.0, 3.0, 4.0]

    def test_single_element(self) -> None:
        assert _first_difference([5.0]) == []

    def test_empty(self) -> None:
        assert _first_difference([]) == []

    def test_constant(self) -> None:
        assert _first_difference([1.0, 1.0, 1.0]) == [0.0, 0.0]


class TestDetrendByBaseline:
    def test_basic(self) -> None:
        series = [0.5, 0.6, 0.7]
        baseline = [0.3, 0.3, 0.3]
        result = _detrend_by_baseline(series, baseline)
        assert result == pytest.approx([0.2, 0.3, 0.4])

    def test_none_baseline(self) -> None:
        series = [0.5, 0.6]
        assert _detrend_by_baseline(series, None) == series

    def test_short_baseline_extends(self) -> None:
        """Baseline shorter than series uses last value for excess positions."""
        series = [0.5, 0.6, 0.7, 0.8]
        baseline = [0.3, 0.3]
        result = _detrend_by_baseline(series, baseline)
        assert result == pytest.approx([0.2, 0.3, 0.4, 0.5])


class TestSegment:
    def test_three_segments(self) -> None:
        series = list(range(9))
        fracs = ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))
        segments = _segment(series, fracs)
        assert segments[0] == [0, 1, 2]
        assert segments[1] == [3, 4, 5]
        assert segments[2] == [6, 7, 8]

    def test_short_series(self) -> None:
        series = [1.0, 2.0]
        fracs = ((0.0, 0.5), (0.5, 1.0))
        segments = _segment(series, fracs)
        assert segments[0] == [1.0]
        assert segments[1] == [2.0]


# -------------------------------------------------------------------
# PEConfig
# -------------------------------------------------------------------


class TestPEConfig:
    def test_default_min_length(self) -> None:
        cfg = PEConfig()
        # D=3, tau=1, min_windows = 2 * 3! = 12
        # min_length = 12 + (3-1)*1 = 14
        assert cfg.min_sequence_length == 14

    def test_custom_factor(self) -> None:
        cfg = PEConfig(min_windows_factor=3)
        # 3 * 6 = 18 windows, + 2 = 20
        assert cfg.min_sequence_length == 20


# -------------------------------------------------------------------
# compute_head_pe
# -------------------------------------------------------------------


class TestComputeHeadPE:
    def test_ineligible_short_sequence(self) -> None:
        cfg = PEConfig()  # min_length = 14
        result = compute_head_pe(
            [0.1, 0.2, 0.3],
            layer_idx=0,
            head_idx=0,
            mode="raw",
            segment="full",
            config=cfg,
        )
        assert not result.eligible
        assert result.pe is None
        assert result.sequence_length == 3

    def test_eligible_sequence(self) -> None:
        # 20 points with some variation
        series = [float(i % 5) * 0.1 + i * 0.01 for i in range(20)]
        cfg = PEConfig()
        result = compute_head_pe(
            series,
            layer_idx=0,
            head_idx=0,
            mode="raw",
            segment="full",
            config=cfg,
        )
        assert result.eligible
        assert result.pe is not None
        assert 0.0 <= result.pe <= 1.0
        assert result.n_strict_patterns > 0
        assert result.sequence_length == 20

    def test_constant_series_high_tie_rate(self) -> None:
        """Constant series should have high tie rate, pe=None."""
        series = [0.5] * 20
        cfg = PEConfig()
        result = compute_head_pe(
            series,
            layer_idx=0,
            head_idx=0,
            mode="raw",
            segment="full",
            config=cfg,
        )
        assert result.eligible
        assert result.pe is None  # all windows tied
        assert result.tie_rate == pytest.approx(1.0)
        assert result.n_strict_patterns == 0

    def test_metadata_preserved(self) -> None:
        series = [float(i) for i in range(20)]
        cfg = PEConfig()
        result = compute_head_pe(
            series,
            layer_idx=5,
            head_idx=12,
            mode="diff",
            segment="early",
            config=cfg,
        )
        assert result.layer_idx == 5
        assert result.head_idx == 12
        assert result.mode == "diff"
        assert result.segment == "early"


# -------------------------------------------------------------------
# compute_sample_pe_features
# -------------------------------------------------------------------


class TestComputeSamplePEFeatures:
    def _make_per_step(self, n_steps: int, n_layers: int, n_heads: int) -> list[dict]:
        """Generate synthetic per-step records."""
        records = []
        for step in range(n_steps):
            for layer in range(n_layers):
                deltas = [
                    0.2 + 0.01 * step + 0.001 * layer + 0.0001 * head for head in range(n_heads)
                ]
                records.append(
                    {
                        "step_idx": step,
                        "layer_idx": layer,
                        "per_head_delta": deltas,
                    }
                )
        return records

    def test_basic_output_shape(self) -> None:
        per_step = self._make_per_step(30, 2, 2)
        result = compute_sample_pe_features(
            per_step,
            num_layers=2,
            num_heads=2,
            dataset_index=42,
        )
        assert result.dataset_index == 42
        # 2 layers * 2 heads * 2 modes (raw, diff) * 4 segments (full, early, mid, late) = 32
        assert len(result.heads) == 32

    def test_modes_control(self) -> None:
        per_step = self._make_per_step(30, 1, 1)
        result = compute_sample_pe_features(
            per_step,
            num_layers=1,
            num_heads=1,
            dataset_index=0,
            modes=("raw",),
            include_segments=False,
        )
        # 1 head * 1 mode * 1 segment (full only) = 1
        assert len(result.heads) == 1
        assert result.heads[0].mode == "raw"
        assert result.heads[0].segment == "full"

    def test_residual_mode_with_baseline(self) -> None:
        per_step = self._make_per_step(30, 1, 1)
        baseline = {(0, 0): [0.2 + 0.01 * i for i in range(30)]}
        result = compute_sample_pe_features(
            per_step,
            num_layers=1,
            num_heads=1,
            dataset_index=0,
            modes=("raw", "diff"),
            baseline=baseline,
            include_segments=False,
        )
        modes = {h.mode for h in result.heads}
        assert "residual" in modes
        assert "raw" in modes
        assert "diff" in modes

    def test_residual_with_partial_baseline_raises(self) -> None:
        """Residual mode with incomplete baseline must raise, not silently use raw."""
        per_step = self._make_per_step(30, 2, 1)
        # Baseline only covers head (0, 0), missing (1, 0)
        baseline = {(0, 0): [0.2 + 0.01 * i for i in range(30)]}
        with pytest.raises(ValueError, match=r"baseline.*missing"):
            compute_sample_pe_features(
                per_step,
                num_layers=2,
                num_heads=1,
                dataset_index=0,
                modes=("raw", "diff"),
                baseline=baseline,
                include_segments=False,
            )

    def test_serialization(self) -> None:
        per_step = self._make_per_step(30, 1, 1)
        result = compute_sample_pe_features(
            per_step,
            num_layers=1,
            num_heads=1,
            dataset_index=7,
            include_segments=False,
            modes=("raw",),
        )
        d = result.to_dict()
        assert d["dataset_index"] == 7
        assert len(d["heads"]) == 1
        assert "pe" in d["heads"][0]
        assert "tie_rate" in d["heads"][0]
        assert "eligible" in d["heads"][0]


# -------------------------------------------------------------------
# compute_positional_baseline
# -------------------------------------------------------------------


class TestComputePositionalBaseline:
    def test_single_sample(self) -> None:
        sample_series = {(0, 0): [0.1, 0.2, 0.3]}
        baseline = compute_positional_baseline([sample_series])
        assert baseline[(0, 0)] == pytest.approx([0.1, 0.2, 0.3])

    def test_two_samples_averaged(self) -> None:
        s1 = {(0, 0): [0.1, 0.2]}
        s2 = {(0, 0): [0.3, 0.4]}
        baseline = compute_positional_baseline([s1, s2])
        assert baseline[(0, 0)] == pytest.approx([0.2, 0.3])

    def test_unequal_lengths(self) -> None:
        """Shorter sample contributes only to positions it covers."""
        s1 = {(0, 0): [0.1, 0.2, 0.3]}
        s2 = {(0, 0): [0.5, 0.6]}
        baseline = compute_positional_baseline([s1, s2])
        # Step 0: mean(0.1, 0.5) = 0.3
        # Step 1: mean(0.2, 0.6) = 0.4
        # Step 2: mean(0.3) = 0.3 (only s1)
        assert baseline[(0, 0)] == pytest.approx([0.3, 0.4, 0.3])
