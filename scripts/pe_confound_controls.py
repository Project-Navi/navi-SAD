#!/usr/bin/env python3
"""PE Confound Controls — thin CLI entry point.

Pure glue: parse args, call tested modules, write outputs.
All preparation, computation, and rendering live in tested modules.

Three analyses:
1. Full-cohort signed asymmetry null
2. Length-matched analysis (pair-restricted null primary, stratified sensitivity)
3. Unanimous-only analysis

Usage:
    python scripts/pe_confound_controls.py [--results-dir PATH] [--n-permutations N]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from navi_sad.analysis.loader import load_reviewer_votes
from navi_sad.analysis.matching import match_by_token_count
from navi_sad.analysis.permutation import run_asymmetry_null, run_paired_asymmetry_null
from navi_sad.analysis.prep import (
    compute_pe_bundle,
    prepare_series_data,
    prepare_series_data_from_subset,
)
from navi_sad.analysis.recurrence import (
    validate_combo_set,
)
from navi_sad.analysis.report import build_provenance, format_confound_controls_markdown
from navi_sad.analysis.selection import select_unanimous
from navi_sad.analysis.types import AsymmetryNullResult, SelectionDiagnostics
from navi_sad.signal.pe_features import PEConfig

logger = logging.getLogger(__name__)

NUM_LAYERS = 32
NUM_HEADS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="PE Confound Controls")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/pilot_gate3_400",
        help="Directory containing samples.json, review.json, and labeling/",
    )
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=2)
    parser.add_argument("--D", type=int, default=3, help="PE embedding dimension")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results_dir = Path(args.results_dir)
    pe_config = PEConfig(D=args.D)

    # ── Shared prep (load, validate, extract series, compute baseline) ──
    logger.info("Preparing full-cohort series data from %s", results_dir)
    series_data = prepare_series_data(results_dir, num_layers=NUM_LAYERS, num_heads=NUM_HEADS)
    logger.info(
        "Loaded %d correct, %d incorrect",
        series_data.input.n_correct,
        series_data.input.n_incorrect,
    )

    logger.info("Computing PE features (D=%d)", pe_config.D)
    pe_bundle = compute_pe_bundle(series_data, pe_config=pe_config)
    validate_combo_set(pe_bundle.lookup)

    provenance = build_provenance(
        series_data.input, pe_config, num_layers=NUM_LAYERS, num_heads=NUM_HEADS
    )

    # ── Analysis 1: Full-cohort signed asymmetry null ──
    logger.info(
        "Analysis 1: Full-cohort signed asymmetry null (%d permutations)", args.n_permutations
    )
    full_cohort_result = run_asymmetry_null(
        lookup=pe_bundle.lookup,
        labels=series_data.input.labels,
        token_counts=series_data.input.token_counts,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        n_permutations=args.n_permutations,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    logger.info(
        "Full-cohort: signed_excess=%d, p_two=%.4f, p_neg=%.4f",
        full_cohort_result.observed.signed_excess,
        full_cohort_result.p_value_two_sided,
        full_cohort_result.p_value_one_sided_negative,
    )

    # ── Analysis 2: Length-matched ──
    logger.info("Analysis 2: Length-matched")
    match_spec, match_diag, pairs = match_by_token_count(
        series_data.input.labels, series_data.input.token_counts
    )
    logger.info(
        "Matched: %d correct + %d incorrect -> %d pairs",
        match_spec.n_correct,
        match_spec.n_incorrect,
        len(pairs),
    )

    matched_result: AsymmetryNullResult | None = None
    matched_sensitivity: AsymmetryNullResult | None = None

    if match_spec.n_correct > 0 and match_spec.n_incorrect > 0:
        # Prepare subset with full-cohort baseline
        matched_series = prepare_series_data_from_subset(
            series_data.input,
            indices=set(match_spec.included_indices),
            baseline=series_data.baseline,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
        )
        matched_bundle = compute_pe_bundle(matched_series, pe_config=pe_config)

        # Matched labels (only paired samples)
        matched_labels = matched_series.input.labels
        matched_token_counts = matched_series.input.token_counts

        # Primary: pair-restricted null
        logger.info("  Pair-restricted null (%d permutations)", args.n_permutations)
        matched_result = run_paired_asymmetry_null(
            lookup=matched_bundle.lookup,
            labels=matched_labels,
            pairs=pairs,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        logger.info(
            "  Paired: signed_excess=%d, p_two=%.4f",
            matched_result.observed.signed_excess,
            matched_result.p_value_two_sided,
        )

        # Sensitivity: stratified null
        logger.info("  Stratified null (sensitivity, %d permutations)", args.n_permutations)
        matched_sensitivity = run_asymmetry_null(
            lookup=matched_bundle.lookup,
            labels=matched_labels,
            token_counts=matched_token_counts,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            n_permutations=args.n_permutations,
            n_bins=1,  # matched subset, stratification not needed
            seed=args.seed,
        )
        logger.info(
            "  Stratified: p_two=%.4f",
            matched_sensitivity.p_value_two_sided,
        )
    else:
        logger.warning("No matched pairs — skipping length-matched analysis")

    # ── Analysis 3: Unanimous-only ──
    logger.info("Analysis 3: Unanimous-only")
    labeling_dir = results_dir / "labeling"
    unanimous_result: AsymmetryNullResult | None = None
    unanimous_diag: SelectionDiagnostics | None = None

    if labeling_dir.exists():
        votes = load_reviewer_votes(labeling_dir)
        unan_spec, unanimous_diag = select_unanimous(votes, series_data.input.labels)
        logger.info(
            "Unanimous: %d correct + %d incorrect (excluded %d ambiguous, %d non-unanimous)",
            unan_spec.n_correct,
            unan_spec.n_incorrect,
            unanimous_diag.n_excluded_ambiguous,
            unanimous_diag.n_excluded_non_unanimous,
        )

        if unan_spec.n_correct > 0 and unan_spec.n_incorrect > 0:
            unan_series = prepare_series_data_from_subset(
                series_data.input,
                indices=set(unan_spec.included_indices),
                baseline=series_data.baseline,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
            )
            unan_bundle = compute_pe_bundle(unan_series, pe_config=pe_config)

            logger.info("  Asymmetry null (%d permutations)", args.n_permutations)
            unanimous_result = run_asymmetry_null(
                lookup=unan_bundle.lookup,
                labels=unan_series.input.labels,
                token_counts=unan_series.input.token_counts,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                n_permutations=args.n_permutations,
                n_bins=args.n_bins,
                seed=args.seed,
            )
            logger.info(
                "  Unanimous: signed_excess=%d, p_two=%.4f",
                unanimous_result.observed.signed_excess,
                unanimous_result.p_value_two_sided,
            )
        else:
            logger.warning("No unanimous correct+incorrect — skipping analysis")
    else:
        logger.warning("No labeling/ directory — skipping unanimous analysis")

    # ── Write outputs ──
    # JSON
    json_data: dict = {
        "provenance": provenance,
        "full_cohort": full_cohort_result.to_dict(),
    }
    if matched_result is not None:
        json_data["length_matched"] = {
            "diagnostics": match_diag.to_dict(),
            "paired_null": matched_result.to_dict(),
        }
        if matched_sensitivity is not None:
            json_data["length_matched"]["stratified_null_sensitivity"] = (
                matched_sensitivity.to_dict()
            )
    if unanimous_result is not None and unanimous_diag is not None:
        json_data["unanimous_only"] = {
            "diagnostics": unanimous_diag.to_dict(),
            "null": unanimous_result.to_dict(),
        }

    json_path = results_dir / "pe_confound_controls.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    logger.info("Wrote %s", json_path)

    # Markdown
    md = format_confound_controls_markdown(
        full_cohort=full_cohort_result,
        matched_result=matched_result,
        matched_diag=match_diag,
        matched_sensitivity=matched_sensitivity,
        unanimous_result=unanimous_result,
        unanimous_diag=unanimous_diag,
        provenance=provenance,
    )
    md_path = results_dir / "pe_confound_controls.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Wrote %s", md_path)

    # ── Summary ──
    logger.info("=== Summary ===")
    logger.info(
        "Full-cohort: signed_excess=%d, p_two=%.4f",
        full_cohort_result.observed.signed_excess,
        full_cohort_result.p_value_two_sided,
    )
    if matched_result:
        logger.info(
            "Length-matched (paired): signed_excess=%d, p_two=%.4f",
            matched_result.observed.signed_excess,
            matched_result.p_value_two_sided,
        )
    if unanimous_result:
        logger.info(
            "Unanimous-only: signed_excess=%d, p_two=%.4f",
            unanimous_result.observed.signed_excess,
            unanimous_result.p_value_two_sided,
        )


if __name__ == "__main__":
    main()
