"""SAD-specific temporal PE features.

Wraps the generic ordinal.py PE engine with SAD-aware sequence
preparation: per-(layer, head) extraction, first-differencing,
detrending, segmentation, and eligibility gating.

Does NOT modify ordinal.py. The surgery is in what sequence we
feed it and how we package the result.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from navi_sad.signal.ordinal import permutation_entropy, recommended_min_pe_length

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------


@dataclass(frozen=True)
class PEConfig:
    """Configuration for SAD temporal PE computation."""

    D: int = 3
    tau: int = 1
    epsilon: float = 1e-9
    # SAD-specific: stricter than the generic D! default.
    # 2 * D! windows ensures the distribution is better sampled.
    min_windows_factor: int = 2
    # Segment boundaries as fractions of sequence length.
    # (0.0, 0.33), (0.33, 0.67), (0.67, 1.0) for early/mid/late.
    segment_fractions: tuple[tuple[float, float], ...] = (
        (0.0, 1.0 / 3),
        (1.0 / 3, 2.0 / 3),
        (2.0 / 3, 1.0),
    )
    segment_names: tuple[str, ...] = ("early", "mid", "late")

    @property
    def min_sequence_length(self) -> int:
        """Minimum sequence length for full-sequence PE."""
        min_windows = self.min_windows_factor * math.factorial(self.D)
        return recommended_min_pe_length(self.D, self.tau, min_windows)


# -------------------------------------------------------------------
# Result types
# -------------------------------------------------------------------


@dataclass(frozen=True)
class HeadPEResult:
    """PE features for one (layer, head) pair on one sequence mode."""

    layer_idx: int
    head_idx: int
    mode: str  # "raw", "diff", "residual"
    segment: str  # "full", "early", "mid", "late"
    sequence_length: int
    eligible: bool  # met minimum length threshold
    pe: float | None  # None if ineligible or all tied
    tie_rate: float
    n_strict_patterns: int
    pattern_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class SamplePEFeatures:
    """All PE features for one sample."""

    dataset_index: int
    config: PEConfig
    heads: list[HeadPEResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "dataset_index": self.dataset_index,
            "config": {
                "D": self.config.D,
                "tau": self.config.tau,
                "min_windows_factor": self.config.min_windows_factor,
            },
            "heads": [
                {
                    "layer_idx": h.layer_idx,
                    "head_idx": h.head_idx,
                    "mode": h.mode,
                    "segment": h.segment,
                    "sequence_length": h.sequence_length,
                    "eligible": h.eligible,
                    "pe": h.pe,
                    "tie_rate": h.tie_rate,
                    "n_strict_patterns": h.n_strict_patterns,
                }
                for h in self.heads
            ],
        }


# -------------------------------------------------------------------
# Sequence extraction
# -------------------------------------------------------------------


def extract_head_sad_series(
    per_step: list[dict[str, Any]],
    num_layers: int,
    num_heads: int,
) -> dict[tuple[int, int], list[float]]:
    """Extract per-(layer, head) SAD delta time series from per-step records.

    Each step may have one record per layer. For each (layer, head),
    collects the delta values ordered by step_idx to form the temporal
    series.

    Returns:
        Dict mapping (layer_idx, head_idx) to list of delta values
        ordered by generation step.
    """
    # Group by (layer, step) to handle multi-layer records per step
    by_layer_step: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for rec in per_step:
        layer = rec["layer_idx"]
        step = rec["step_idx"]
        by_layer_step[layer][step] = rec["per_head_delta"]

    result: dict[tuple[int, int], list[float]] = {}
    for layer_idx in range(num_layers):
        step_data = by_layer_step.get(layer_idx, {})
        steps = sorted(step_data.keys())
        for head_idx in range(num_heads):
            series = [step_data[s][head_idx] for s in steps]
            result[(layer_idx, head_idx)] = series

    return result


# -------------------------------------------------------------------
# Sequence transforms
# -------------------------------------------------------------------


def _first_difference(series: list[float]) -> list[float]:
    """Compute first differences: series[t] - series[t-1]."""
    if len(series) < 2:
        return []
    return [series[i] - series[i - 1] for i in range(1, len(series))]


def _detrend_by_baseline(
    series: list[float],
    baseline: list[float] | None,
) -> list[float]:
    """Subtract a positional baseline from the series.

    If baseline is shorter than series, excess positions use the
    last baseline value. If baseline is None, returns series unchanged.
    """
    if baseline is None:
        return series
    result: list[float] = []
    for i, val in enumerate(series):
        base = baseline[i] if i < len(baseline) else baseline[-1]
        result.append(val - base)
    return result


def _segment(
    series: list[float],
    fractions: tuple[tuple[float, float], ...],
) -> list[list[float]]:
    """Split series into segments defined by fractional boundaries."""
    n = len(series)
    segments: list[list[float]] = []
    for start_frac, end_frac in fractions:
        start_idx = int(n * start_frac)
        end_idx = int(n * end_frac)
        segments.append(series[start_idx:end_idx])
    return segments


# -------------------------------------------------------------------
# Core PE computation
# -------------------------------------------------------------------


def compute_head_pe(
    series: list[float],
    *,
    layer_idx: int,
    head_idx: int,
    mode: str,
    segment: str,
    config: PEConfig,
) -> HeadPEResult:
    """Compute PE for a single (layer, head) series.

    Applies eligibility gating based on config.min_sequence_length.
    Returns a HeadPEResult with all metadata.
    """
    seq_len = len(series)
    min_len = config.min_sequence_length

    if seq_len < min_len:
        return HeadPEResult(
            layer_idx=layer_idx,
            head_idx=head_idx,
            mode=mode,
            segment=segment,
            sequence_length=seq_len,
            eligible=False,
            pe=None,
            tie_rate=0.0,
            n_strict_patterns=0,
        )

    pe_val, tie_rate, pattern_counts = permutation_entropy(
        series, D=config.D, tau=config.tau, epsilon=config.epsilon
    )

    n_strict = sum(pattern_counts.values())

    return HeadPEResult(
        layer_idx=layer_idx,
        head_idx=head_idx,
        mode=mode,
        segment=segment,
        sequence_length=seq_len,
        eligible=True,
        pe=pe_val,
        tie_rate=tie_rate,
        n_strict_patterns=n_strict,
        pattern_counts=dict(pattern_counts),
    )


# -------------------------------------------------------------------
# Sample-level batch computation
# -------------------------------------------------------------------


def compute_sample_pe_features(
    per_step: list[dict[str, Any]],
    num_layers: int,
    num_heads: int,
    dataset_index: int,
    *,
    config: PEConfig | None = None,
    baseline: dict[tuple[int, int], list[float]] | None = None,
    modes: tuple[str, ...] = ("raw", "diff"),
    include_segments: bool = True,
) -> SamplePEFeatures:
    """Compute PE features for all (layer, head) pairs in a sample.

    Args:
        per_step: Per-step records from the sample.
        num_layers: Number of model layers.
        num_heads: Number of attention heads per layer.
        dataset_index: Sample identifier.
        config: PE configuration. Defaults to PEConfig().
        baseline: Optional per-(layer, head) positional baseline
            for residual mode. If provided, "residual" is added
            to modes automatically.
        modes: Which sequence transforms to compute PE on.
            "raw" = raw SAD series, "diff" = first differences,
            "residual" = detrended by baseline.
        include_segments: Whether to compute segment-wise PE
            (early/mid/late) in addition to full-sequence PE.

    Returns:
        SamplePEFeatures with all computed head PE results.
    """
    if config is None:
        config = PEConfig()

    head_series = extract_head_sad_series(per_step, num_layers, num_heads)
    active_modes = list(modes)
    if baseline is not None and "residual" not in active_modes:
        active_modes.append("residual")

    results: list[HeadPEResult] = []

    for (layer_idx, head_idx), raw_series in head_series.items():
        # Build transformed series for each mode
        mode_series: dict[str, list[float]] = {}
        if "raw" in active_modes:
            mode_series["raw"] = raw_series
        if "diff" in active_modes:
            mode_series["diff"] = _first_difference(raw_series)
        if "residual" in active_modes and baseline is not None:
            head_baseline = baseline.get((layer_idx, head_idx))
            mode_series["residual"] = _detrend_by_baseline(raw_series, head_baseline)

        for mode_name, series in mode_series.items():
            # Full sequence
            results.append(
                compute_head_pe(
                    series,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    mode=mode_name,
                    segment="full",
                    config=config,
                )
            )

            # Segments
            if include_segments:
                segments = _segment(series, config.segment_fractions)
                for seg_series, seg_name in zip(segments, config.segment_names, strict=True):
                    results.append(
                        compute_head_pe(
                            seg_series,
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            mode=mode_name,
                            segment=seg_name,
                            config=config,
                        )
                    )

    return SamplePEFeatures(
        dataset_index=dataset_index,
        config=config,
        heads=results,
    )


# -------------------------------------------------------------------
# Baseline computation
# -------------------------------------------------------------------


def compute_positional_baseline(
    all_head_series: list[dict[tuple[int, int], list[float]]],
) -> dict[tuple[int, int], list[float]]:
    """Compute mean SAD delta at each step position across samples.

    For each (layer, head), averages the delta at each step_idx
    across all samples that have data at that position. This serves
    as the positional baseline for residual detrending.

    Args:
        all_head_series: List of per-sample head series dicts
            (output of extract_head_sad_series for each sample).

    Returns:
        Dict mapping (layer_idx, head_idx) to baseline series.
    """
    # Collect all values by (layer, head, step)
    accum: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for sample_series in all_head_series:
        for (layer_idx, head_idx), series in sample_series.items():
            for step_idx, val in enumerate(series):
                accum[(layer_idx, head_idx)][step_idx].append(val)

    # Average
    baseline: dict[tuple[int, int], list[float]] = {}
    for key, step_data in accum.items():
        max_step = max(step_data.keys())
        means: list[float] = []
        for step in range(max_step + 1):
            vals = step_data.get(step, [])
            means.append(sum(vals) / len(vals) if vals else 0.0)
        baseline[key] = means

    return baseline
