#!/usr/bin/env python3
"""PE Recurrence Null Test — thin CLI entry point.

Pure glue: parse args, call tested modules, write outputs.
All preparation, computation, and rendering live in tested modules.

Usage:
    python scripts/pe_recurrence_null.py [--results-dir PATH] [--n-permutations N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.prep import compute_pe_bundle, prepare_series_data
from navi_sad.analysis.recurrence import (
    compute_d_matrix,
    summarize_d_matrix,
    validate_combo_set,
)
from navi_sad.analysis.report import build_provenance, format_markdown
from navi_sad.analysis.types import PermutationNullConfig, RecurrenceNullReport
from navi_sad.signal.pe_features import PEConfig

log = structlog.get_logger()

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
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON log lines")
    args = parser.parse_args()

    from navi_sad.logging import configure_logging

    configure_logging(json=args.json_logs)

    results_dir = Path(args.results_dir)

    # Prepare (load, validate, extract series, compute baseline)
    log.info("Preparing series data from %s", results_dir)
    series_data = prepare_series_data(results_dir, num_layers=NUM_LAYERS, num_heads=NUM_HEADS)
    log.info(
        "Included: %d correct, %d incorrect",
        series_data.input.n_correct,
        series_data.input.n_incorrect,
    )

    # Compute PE features at D=3
    log.info("Computing PE bundle (D=3)...")
    pe_config = PEConfig()
    bundle = compute_pe_bundle(series_data, pe_config)

    # Validate 12-combo contract
    validate_combo_set(bundle.lookup)

    # Compute d matrix (never discard d values)
    log.info("Computing d matrix...")
    d_matrix = compute_d_matrix(
        bundle.lookup, series_data.input.labels, num_layers=NUM_LAYERS, num_heads=NUM_HEADS
    )
    d_landscape = summarize_d_matrix(d_matrix, num_layers=NUM_LAYERS, num_heads=NUM_HEADS)
    log.info(
        "d landscape: %d/%d cells present, max|d|=%.4f, mean|d|=%.4f, positive=%.1f%%",
        d_landscape.present_cells,
        d_landscape.expected_total_cells,
        d_landscape.max_abs_d or 0,
        d_landscape.mean_abs_d or 0,
        (d_landscape.positive_fraction or 0) * 100,
    )

    # Permutation null
    config = PermutationNullConfig(
        n_permutations=args.n_permutations,
        seed=args.seed,
        n_bins=args.n_bins,
    )
    log.info(
        "Running permutation null: %d permutations, %d bins, seed=%d...",
        config.n_permutations,
        config.n_bins,
        config.seed,
    )
    report = run_permutation_null(
        bundle.lookup,
        series_data.input.labels,
        series_data.input.token_counts,
        config=config,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    # Attach eligibility and d landscape (construct new frozen report)
    report = RecurrenceNullReport(
        config=report.config,
        eligibility=bundle.eligibility,
        observed=report.observed,
        observed_profile=report.observed_profile,
        null_at_min_combos=report.null_at_min_combos,
        null_at_seven=report.null_at_seven,
        bin_boundaries=report.bin_boundaries,
        bin_counts=report.bin_counts,
        d_landscape=d_landscape,
    )

    # Provenance + output
    provenance = build_provenance(series_data.input, pe_config, NUM_LAYERS, NUM_HEADS)

    json_path = results_dir / "pe_recurrence_null.json"
    report_dict = report.to_dict()
    report_dict["provenance"] = provenance
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    log.info("Wrote %s", json_path)

    md_path = results_dir / "pe_recurrence_null.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(format_markdown(report, provenance))
    log.info("Wrote %s", md_path)

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
