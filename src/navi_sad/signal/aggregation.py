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

    max_step = max(by_step.keys()) if by_step else -1
    result = []
    for i in range(max_step + 1):
        values = by_step.get(i, [])
        result.append(sum(values) / len(values) if values else 0.0)
    return result
