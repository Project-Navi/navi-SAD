"""PE data preparation for analysis.

Two-layer prep: series-level (D-independent) then PE-level (D-dependent).
All PE preparation logic lives here, not in CLI scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import AnalysisInput, load_and_validate, step_records_to_dicts
from navi_sad.analysis.recurrence import PELookup, build_pe_lookup
from navi_sad.analysis.types import EligibilityTable
from navi_sad.signal.pe_features import (
    PEConfig,
    SamplePEFeatures,
    compute_positional_baseline,
    compute_sample_pe_features,
    extract_head_sad_series,
)


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

    return SeriesData(
        input=data,
        head_series=head_series,
        baseline=baseline,
        per_step_dicts=per_step_dicts,
        num_layers=num_layers,
        num_heads=num_heads,
    )


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

    return PEBundle(
        pe_samples=pe_samples,
        lookup=lookup,
        eligibility=eligibility,
        pe_config=pe_config,
    )
