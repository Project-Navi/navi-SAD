#!/usr/bin/env python3
"""PE Recurrence Null Test — thin CLI entry point.

Loads pilot artifacts, computes PE features, calls analysis
modules, writes structured results. Zero analysis logic here.

Usage:
    python scripts/pe_recurrence_null.py [--results-dir PATH] [--n-permutations N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.recurrence import build_pe_lookup
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
    samples_path = results_dir / "samples.json"
    review_path = results_dir / "review.json"

    if not samples_path.exists() or not review_path.exists():
        sys.exit(f"Missing artifacts in {results_dir}")

    # Load
    logger.info("Loading artifacts from %s", results_dir)
    with open(samples_path, encoding="utf-8") as f:
        samples_artifact: dict[str, Any] = json.load(f)
    with open(review_path, encoding="utf-8") as f:
        review_data: list[dict[str, Any]] = json.load(f)

    samples_raw = samples_artifact["samples"]
    labels_raw = {r["dataset_index"]: r["human_label"] for r in review_data}

    # Filter: correct/incorrect only, no sample errors
    included = [
        s
        for s in samples_raw
        if labels_raw.get(s["dataset_index"]) in ("correct", "incorrect")
        and s.get("sample_error") is None
    ]
    labels = {s["dataset_index"]: labels_raw[s["dataset_index"]] for s in included}
    token_counts = {s["dataset_index"]: s["generated_token_count"] for s in included}

    n_correct = sum(1 for v in labels.values() if v == "correct")
    n_incorrect = sum(1 for v in labels.values() if v == "incorrect")
    logger.info("Included: %d correct, %d incorrect", n_correct, n_incorrect)

    # Compute PE features
    logger.info("Computing PE features (3 modes x 4 segments)...")
    pe_config = PEConfig()

    # Extract head series for baseline
    all_head_series = []
    for s in included:
        hs = extract_head_sad_series(s["per_step"], NUM_LAYERS, NUM_HEADS)
        all_head_series.append(hs)

    baseline = compute_positional_baseline(all_head_series)

    pe_samples = {}
    for s in included:
        idx = s["dataset_index"]
        pe = compute_sample_pe_features(
            s["per_step"],
            NUM_LAYERS,
            NUM_HEADS,
            idx,
            config=pe_config,
            baseline=baseline,
            modes=("raw", "diff"),
            include_segments=True,
        )
        pe_samples[idx] = pe

    # Eligibility
    logger.info("Building eligibility table...")
    eligibility = build_eligibility_table(pe_samples, labels)

    # PE lookup
    lookup = build_pe_lookup(pe_samples)

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
        labels,
        token_counts,
        config=config,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    )

    # Attach eligibility (computed separately)
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

    # Write JSON (to_dict() already excludes null_counts — only summary stats)
    json_path = results_dir / "pe_recurrence_null.json"
    report_dict = report.to_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Wrote %s", json_path)

    # Write markdown
    md_path = results_dir / "pe_recurrence_null.md"
    md = format_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Wrote %s", md_path)

    # Summary to stdout
    print(
        f"\nObserved recurring heads (>={config.min_combos} combos): "
        f"{report.observed.recurring_head_count}"
    )
    print(f"Expected under null: {report.null_at_min_combos.expected_under_null:.1f}")
    print(f"p-value (>={config.min_combos} combos): {report.null_at_min_combos.p_value:.4f}")
    print(f"p-value (>=7 combos): {report.null_at_seven.p_value:.4f}")


def format_markdown(report: RecurrenceNullReport) -> str:
    """Render report as markdown. Eligibility tables first."""
    lines: list[str] = []
    lines.append("# PE Recurrence Null Test Results\n")

    # Eligibility
    lines.append("## Eligibility by Class x Mode x Segment\n")
    if report.eligibility is not None:
        lines.append(
            f"Samples: {report.eligibility.n_correct} correct, "
            f"{report.eligibility.n_incorrect} incorrect\n"
        )
        lines.append(
            "| Mode | Segment | Correct Eligible | Correct PE-present "
            "| Incorrect Eligible | Incorrect PE-present |"
        )
        lines.append(
            "|------|---------|------------------|--------------------"
            "|--------------------|----------------------|"
        )
        for c in report.eligibility.cells:
            lines.append(
                f"| {c.mode} | {c.segment} | "
                f"{c.n_correct_eligible}/{c.n_correct_total} | "
                f"{c.n_correct_pe_present}/{c.n_correct_total} | "
                f"{c.n_incorrect_eligible}/{c.n_incorrect_total} | "
                f"{c.n_incorrect_pe_present}/{c.n_incorrect_total} |"
            )
        lines.append("")

    # Observed
    lines.append("## Observed Recurrence\n")
    lines.append(
        f"- **Test statistic:** heads with |d| > {report.observed.d_threshold} "
        f"in >= {report.observed.min_combos} combos"
    )
    lines.append(
        f"- **Observed count:** {report.observed.recurring_head_count} "
        f"/ {report.observed.total_heads}"
    )
    lines.append("")

    # Profile
    lines.append("### Recurrence Profile\n")
    lines.append("| Min Combos | Heads >= |")
    lines.append("|-----------|----------|")
    for level, count in sorted(report.observed_profile.counts_at_level.items()):
        marker = " **" if level in (report.config.min_combos, 7) else ""
        lines.append(f"| >= {level} | {count}{marker}")
    lines.append("")

    # Null test
    lines.append("## Permutation Null Test\n")
    lines.append(f"- **N permutations:** {report.config.n_permutations}")
    lines.append(
        f"- **Stratification:** {report.config.n_bins} bins, boundaries={report.bin_boundaries}"
    )
    lines.append(f"- **Seed:** {report.config.seed}")
    lines.append("")

    null_min = report.null_at_min_combos
    lines.append(f"### At >= {report.config.min_combos} combos\n")
    lines.append(f"- Observed: {null_min.observed}")
    lines.append(f"- Expected under null: {null_min.expected_under_null:.1f}")
    lines.append(f"- **p-value: {null_min.p_value:.4f}**")
    lines.append(f"- Null range: [{null_min.null_min}, {null_min.null_max}]")
    lines.append(f"- Null percentiles: {null_min.null_percentiles}")
    lines.append("")

    null_7 = report.null_at_seven
    lines.append("### At >= 7 combos\n")
    lines.append(f"- Observed: {null_7.observed}")
    lines.append(f"- Expected under null: {null_7.expected_under_null:.1f}")
    lines.append(f"- **p-value: {null_7.p_value:.4f}**")
    lines.append(f"- Null range: [{null_7.null_min}, {null_7.null_max}]")
    lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    lines.append(
        "- **GQA non-independence:** Mistral uses 8 KV groups with 32 Q heads. "
        "The 1024 head-level tests are not independent. "
        "Grouped/cluster-aware analysis is a separate follow-up."
    )
    lines.append("- **Small n:** 9 incorrect vs 28 correct. All effect sizes are exploratory.")
    lines.append(
        "- **Transform-family dependence:** Raw, diff, and residual modes "
        "are transforms of the same series. Cross-mode recurrence is "
        "robustness evidence, not independent replication."
    )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
