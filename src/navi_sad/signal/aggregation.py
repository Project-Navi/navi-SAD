"""Aggregate per-layer-per-head deltas to per-token delta."""

from collections import defaultdict

from navi_sad.core.types import StepRecord


def aggregate_deltas(
    steps: list[StepRecord],
    method: str = "uniform_mean",
) -> list[float]:
    """Aggregate step records into a per-token delta series.

    Groups by step_idx, then averages all per_head_delta values
    across all layers and heads (uniform mean).

    Args:
        steps: list of StepRecord from raw inference
        method: aggregation method (only "uniform_mean" supported)

    Returns:
        list of float, one per generation step
    """
    if method != "uniform_mean":
        raise ValueError(f"Unknown aggregation method: {method}")

    by_step: dict[int, list[float]] = defaultdict(list)
    for s in steps:
        by_step[s.step_idx].extend(s.per_head_delta)

    if not by_step:
        return []

    max_step = max(by_step.keys())
    expected = set(range(max_step + 1))
    actual = set(by_step.keys())
    missing = expected - actual
    if missing:
        raise ValueError(
            f"non-contiguous step_idx: missing {sorted(missing)}. "
            f"This indicates a step-accounting bug — every step_idx from 0 to "
            f"{max_step} must be present."
        )

    return [sum(by_step[i]) / len(by_step[i]) for i in range(max_step + 1)]
