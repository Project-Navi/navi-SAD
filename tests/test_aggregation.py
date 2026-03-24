"""Tests for per-layer-per-head delta aggregation."""

import math

import pytest

from navi_sad.core.types import StepRecord
from navi_sad.signal.aggregation import aggregate_deltas


# ===========================================================================
# TestAggregateDeltas
# ===========================================================================
class TestAggregateDeltas:
    def test_uniform_mean_single_step(self) -> None:
        """Two StepRecords at step_idx=0 with different layers -> mean of all values."""
        steps = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1, 0.2]),
            StepRecord(step_idx=0, layer_idx=1, per_head_delta=[0.3, 0.4]),
        ]
        result = aggregate_deltas(steps)
        assert len(result) == 1
        expected = (0.1 + 0.2 + 0.3 + 0.4) / 4
        assert math.isclose(result[0], expected, rel_tol=1e-9)

    def test_multiple_steps(self) -> None:
        """Two StepRecords at step_idx=0 and step_idx=1 -> two output values."""
        steps = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1, 0.2]),
            StepRecord(step_idx=1, layer_idx=0, per_head_delta=[0.5, 0.6]),
        ]
        result = aggregate_deltas(steps)
        assert len(result) == 2
        assert math.isclose(result[0], 0.15, rel_tol=1e-9)
        assert math.isclose(result[1], 0.55, rel_tol=1e-9)

    def test_empty(self) -> None:
        """Empty step list should produce empty result."""
        result = aggregate_deltas([])
        assert result == []

    def test_unknown_method_raises(self) -> None:
        """Unknown aggregation method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_deltas([], method="unknown")

    def test_multi_layer_multi_head(self) -> None:
        """2 layers x 4 heads at step 0 -> mean of 8 values."""
        steps = [
            StepRecord(
                step_idx=0,
                layer_idx=0,
                per_head_delta=[0.1, 0.2, 0.3, 0.4],
            ),
            StepRecord(
                step_idx=0,
                layer_idx=1,
                per_head_delta=[0.5, 0.6, 0.7, 0.8],
            ),
        ]
        result = aggregate_deltas(steps)
        assert len(result) == 1
        expected = sum([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) / 8
        assert math.isclose(result[0], expected, rel_tol=1e-9)
