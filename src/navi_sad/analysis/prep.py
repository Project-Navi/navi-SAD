"""PE data preparation for analysis.

Two-layer prep: series-level (D-independent) then PE-level (D-dependent).
All PE preparation logic lives here, not in CLI scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import AnalysisInput, load_and_validate, step_records_to_dicts
from navi_sad.analysis.recurrence import PELookup, build_pe_lookup
from navi_sad.analysis.types import BaselineDeviation, EligibilityTable
from navi_sad.signal.pe_features import (
    PEConfig,
    SamplePEFeatures,
    compute_positional_baseline,
    compute_sample_pe_features,
    extract_head_sad_series,
)

log = structlog.get_logger()


@dataclass(frozen=True)
class SeriesData:
    """D-independent series-level data. Computed once, reused across D values.

    Contains validated input, pre-extracted head series, and the
    shared positional baseline. The per_step_dicts field holds the
    dict-form per-step data needed by the PE API.
    """

    input: AnalysisInput
    head_series: dict[int, dict[tuple[int, int], list[float]]]
    baseline: dict[tuple[int, int], list[float]]
    per_step_dicts: dict[int, list[dict[str, Any]]]
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class PEBundle:
    """PE features at a specific D value. D-dependent, computed per D."""

    pe_samples: dict[int, SamplePEFeatures]
    lookup: PELookup
    eligibility: EligibilityTable
    pe_config: PEConfig


def prepare_series_data(
    results_dir: Path,
    num_layers: int = 32,
    num_heads: int = 32,
) -> SeriesData:
    """Load, validate, extract head series, compute baseline.

    This is the expensive shared step that all analyses reuse.
    D-independent: call once, then compute_pe_bundle() per D value.
    """
    data = load_and_validate(results_dir)

    # Convert StepRecords to dicts for the PE API
    per_step_dicts = {
        idx: step_records_to_dicts(records) for idx, records in data.per_step_data.items()
    }

    # Extract head series for baseline
    head_series: dict[int, dict[tuple[int, int], list[float]]] = {}
    all_head_series = []
    for idx in sorted(per_step_dicts):
        hs = extract_head_sad_series(per_step_dicts[idx], num_layers, num_heads)
        head_series[idx] = hs
        all_head_series.append(hs)

    baseline = compute_positional_baseline(all_head_series)

    log.info(
        "series_data_prepared",
        n_samples=len(head_series),
        num_layers=num_layers,
        num_heads=num_heads,
    )

    return SeriesData(
        input=data,
        head_series=head_series,
        baseline=baseline,
        per_step_dicts=per_step_dicts,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def prepare_series_data_from_subset(
    data: AnalysisInput,
    indices: set[int],
    baseline: dict[tuple[int, int], list[float]],
    num_layers: int,
    num_heads: int,
) -> SeriesData:
    """Build SeriesData from an already-loaded subset, using a provided baseline.

    Filters the AnalysisInput to the given indices. Does NOT recompute
    the positional baseline — uses the full-cohort baseline passed in.
    Skips disk I/O entirely.

    Args:
        data: Already-loaded and validated AnalysisInput.
        indices: Which dataset indices to include.
        baseline: Pre-computed positional baseline (full-cohort).
        num_layers: Number of model layers.
        num_heads: Number of attention heads per layer.

    Raises:
        ValueError: If indices is empty or contains indices not in data.
    """
    if not indices:
        raise ValueError("Empty indices set — no samples to prepare")

    available = set(data.per_step_data.keys())
    missing = indices - available
    if missing:
        raise ValueError(
            f"Indices {sorted(missing)} not in loaded data. Available: {sorted(available)}"
        )

    # Filter labels, token_counts, per_step_data
    filtered_labels = {idx: data.labels[idx] for idx in indices}
    filtered_token_counts = {idx: data.token_counts[idx] for idx in indices}
    filtered_per_step = {idx: data.per_step_data[idx] for idx in indices}

    n_correct = sum(1 for v in filtered_labels.values() if v == "correct")
    n_incorrect = sum(1 for v in filtered_labels.values() if v == "incorrect")

    subset_input = AnalysisInput(
        labels=filtered_labels,
        token_counts=filtered_token_counts,
        per_step_data=filtered_per_step,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        samples_path=data.samples_path,
        review_path=data.review_path,
    )

    # Convert StepRecords to dicts for the PE API
    per_step_dicts = {
        idx: step_records_to_dicts(records) for idx, records in filtered_per_step.items()
    }

    # Extract head series (for the subset, but baseline is NOT recomputed)
    head_series: dict[int, dict[tuple[int, int], list[float]]] = {}
    for idx in sorted(per_step_dicts):
        hs = extract_head_sad_series(per_step_dicts[idx], num_layers, num_heads)
        head_series[idx] = hs

    log.info(
        "subset_prepared",
        n_indices=len(indices),
        num_layers=num_layers,
        num_heads=num_heads,
    )

    return SeriesData(
        input=subset_input,
        head_series=head_series,
        baseline=baseline,
        per_step_dicts=per_step_dicts,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def compute_baseline_deviation(
    subset_head_series: dict[int, dict[tuple[int, int], list[float]]],
    full_baseline: dict[tuple[int, int], list[float]],
) -> BaselineDeviation:
    """Compute how much a subset's positional baseline deviates from full cohort.

    Recomputes what the baseline would be for the subset alone, then
    measures the max and mean absolute deviation from the full-cohort
    baseline across all (layer, head, position) combinations.

    This is the diagnostic promised by the spec: if baselines differ
    substantially, the shared-baseline assumption may not be benign.
    """
    if not subset_head_series:
        return BaselineDeviation(
            max_abs_deviation=0.0,
            mean_abs_deviation=0.0,
            n_positions_compared=0,
        )

    # Recompute subset baseline
    subset_baseline = compute_positional_baseline(list(subset_head_series.values()))

    # Compare against full baseline
    all_deviations: list[float] = []
    for head_key, full_series in full_baseline.items():
        subset_series = subset_baseline.get(head_key)
        if subset_series is None:
            continue
        min_len = min(len(full_series), len(subset_series))
        for pos in range(min_len):
            all_deviations.append(abs(full_series[pos] - subset_series[pos]))

    if not all_deviations:
        return BaselineDeviation(
            max_abs_deviation=0.0,
            mean_abs_deviation=0.0,
            n_positions_compared=0,
        )

    result = BaselineDeviation(
        max_abs_deviation=max(all_deviations),
        mean_abs_deviation=sum(all_deviations) / len(all_deviations),
        n_positions_compared=len(all_deviations),
    )

    log.info(
        "baseline_deviation_computed",
        max_abs=result.max_abs_deviation,
        mean_abs=result.mean_abs_deviation,
        n_positions=result.n_positions_compared,
    )

    return result


def compute_pe_bundle(
    series_data: SeriesData,
    pe_config: PEConfig | None = None,
) -> PEBundle:
    """Compute PE features from series data at a specific D.

    Uses compute_sample_pe_features on each sample's per-step data
    with the shared baseline. Returns PE samples, lookup, and eligibility.
    """
    if pe_config is None:
        pe_config = PEConfig()

    pe_samples: dict[int, SamplePEFeatures] = {}
    for idx in sorted(series_data.per_step_dicts):
        pe = compute_sample_pe_features(
            series_data.per_step_dicts[idx],
            series_data.num_layers,
            series_data.num_heads,
            idx,
            config=pe_config,
            baseline=series_data.baseline,
            # "residual" auto-added by compute_sample_pe_features when
            # baseline is provided (pe_features.py:350-351). This gives
            # 3 modes x 4 segments = 12 combos per head.
            modes=("raw", "diff"),
            include_segments=True,
        )
        pe_samples[idx] = pe

    lookup = build_pe_lookup(pe_samples)
    eligibility = build_eligibility_table(pe_samples, series_data.input.labels)

    log.info(
        "pe_bundle_computed",
        D=pe_config.D,
        n_samples=len(pe_samples),
        n_lookup_combos=len(lookup),
    )

    return PEBundle(
        pe_samples=pe_samples,
        lookup=lookup,
        eligibility=eligibility,
        pe_config=pe_config,
    )
