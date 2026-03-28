#!/usr/bin/env python3
"""PE Recurrence Null Test — thin CLI entry point.

Loads validated artifacts, computes PE features, calls analysis
modules, writes structured results. All boundary validation,
provenance, and rendering live in tested modules.

Usage:
    python scripts/pe_recurrence_null.py [--results-dir PATH] [--n-permutations N]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import load_and_validate
from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.recurrence import (
    build_pe_lookup,
    compute_d_matrix,
    summarize_d_matrix,
    validate_combo_set,
)
from navi_sad.analysis.report import build_provenance, format_markdown
from navi_sad.analysis.types import PermutationNullConfig, RecurrenceNullReport
from navi_sad.signal.pe_features import (
    PEConfig,
    compute_positional_baseline,
    compute_sample_pe_features,
    extract_head_sad_series,
)

logger = logging.getLogger(__name__)

NUM_LAYERS = 32
NUM_HEADS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="PE Recurrence Null Test")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/pilot_gate3",
        help="Directory containing samples.json and review.json",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10_000,
        help="Number of permutations (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results_dir = Path(args.results_dir)

    # Load and validate artifacts (rejects on integrity violations)
    logger.info("Loading and validating artifacts from %s", results_dir)
    data = load_and_validate(results_dir)
    logger.info("Included: %d correct, %d incorrect", data.n_correct, data.n_incorrect)

    # Compute PE features
    logger.info("Computing PE features (3 modes x 4 segments)...")
    pe_config = PEConfig()

    all_head_series = []
    for idx in sorted(data.per_step_data):
        hs = extract_head_sad_series(data.per_step_data[idx], NUM_LAYERS, NUM_HEADS)
        all_head_series.append(hs)

    baseline = compute_positional_baseline(all_head_series)

    pe_samples = {}
    for idx in sorted(data.per_step_data):
        pe = compute_sample_pe_features(
            data.per_step_data[idx],
            NUM_LAYERS,
            NUM_HEADS,
            idx,
            config=pe_config,
            baseline=baseline,
            modes=("raw", "diff"),
            include_segments=True,
        )
        pe_samples[idx] = pe

    # Eligibility + PE lookup + 12-combo contract validation
    logger.info("Building eligibility table...")
    eligibility = build_eligibility_table(pe_samples, data.labels)
    lookup = build_pe_lookup(pe_samples)
    validate_combo_set(lookup)

    # Compute and persist the full d matrix (never discard d values)
    logger.info("Computing d matrix...")
    d_matrix = compute_d_matrix(lookup, data.labels, num_layers=NUM_LAYERS, num_heads=NUM_HEADS)
    d_summary = summarize_d_matrix(d_matrix)
    logger.info(
        "d matrix: max|d|=%.4f, mean|d|=%.4f, positive=%.1f%%, threshold sweep: %s",
        d_summary.get("max_abs_d", 0) or 0,
        d_summary.get("mean_abs_d", 0) or 0,
        (d_summary.get("positive_fraction", 0) or 0) * 100,
        d_summary.get("threshold_sweep", {}),
    )

    # Null test
    config = PermutationNullConfig(
        n_permutations=args.n_permutations,
        seed=args.seed,
        n_bins=args.n_bins,
    )
    logger.info(
        "Running permutation null: %d permutations, %d bins, seed=%d...",
        config.n_permutations,
        config.n_bins,
        config.seed,
    )
    report = run_permutation_null(
        lookup,
        data.labels,
        data.token_counts,
        config=config,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    # Attach eligibility (construct new frozen report)
    report = RecurrenceNullReport(
        config=report.config,
        eligibility=eligibility,
        observed=report.observed,
        observed_profile=report.observed_profile,
        null_at_min_combos=report.null_at_min_combos,
        null_at_seven=report.null_at_seven,
        bin_boundaries=report.bin_boundaries,
        bin_counts=report.bin_counts,
    )

    # Provenance + output
    provenance = build_provenance(data, pe_config, NUM_LAYERS, NUM_HEADS)

    json_path = results_dir / "pe_recurrence_null.json"
    report_dict = report.to_dict()
    report_dict["provenance"] = provenance
    report_dict["d_landscape"] = d_summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Wrote %s", json_path)

    md_path = results_dir / "pe_recurrence_null.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown(report, provenance))
    logger.info("Wrote %s", md_path)

    # Summary to stdout
    print(
        f"\nObserved recurring heads (>={config.min_combos} combos): "
        f"{report.observed.recurring_head_count}"
    )
    print(f"Expected under null: {report.null_at_min_combos.expected_under_null:.1f}")
    print(f"p-value (>={config.min_combos} combos): {report.null_at_min_combos.p_value:.4f}")
    print(f"p-value (>=7 combos): {report.null_at_seven.p_value:.4f}")


if __name__ == "__main__":
    main()
