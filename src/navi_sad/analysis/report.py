"""Report rendering for PE recurrence null analysis.

Builds provenance dicts and renders markdown reports.
Tested module — not in the CLI script.
"""

from __future__ import annotations

from typing import Any

import structlog

from navi_sad.analysis.loader import AnalysisInput
from navi_sad.analysis.types import (
    AsymmetryNullResult,
    BaselineDeviation,
    MatchingDiagnostics,
    RecurrenceNullReport,
    SelectionDiagnostics,
)
from navi_sad.signal.pe_features import PEConfig

log = structlog.get_logger()


def build_provenance(
    data: AnalysisInput,
    pe_config: PEConfig,
    num_layers: int,
    num_heads: int,
) -> dict[str, Any]:
    """Build provenance dict for reproducibility.

    Captures artifact paths, PE configuration, grid dimensions,
    and sample counts. Required by the frozen reporting contract.
    """
    return {
        "samples_path": data.samples_path,
        "review_path": data.review_path,
        "pe_config": {
            "D": pe_config.D,
            "tau": pe_config.tau,
            "epsilon": pe_config.epsilon,
            "min_windows_factor": pe_config.min_windows_factor,
            "segment_fractions": pe_config.segment_fractions,
        },
        "num_layers": num_layers,
        "num_heads": num_heads,
        "n_correct": data.n_correct,
        "n_incorrect": data.n_incorrect,
    }


def format_markdown(
    report: RecurrenceNullReport,
    provenance: dict[str, Any],
) -> str:
    """Render report as markdown. Eligibility first, provenance last."""
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
        marker = " <<" if level in (report.config.min_combos, 7) else ""
        lines.append(f"| >= {level} | {count}{marker} |")
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

    # D landscape
    if report.d_landscape is not None:
        dl = report.d_landscape
        lines.append("## D-Value Landscape\n")
        lines.append(
            f"- **Grid:** {dl.expected_total_cells} expected cells, "
            f"{dl.present_cells} present, {dl.absent_cells} absent"
        )
        if dl.max_abs_d is not None:
            lines.append(f"- **Max |d|:** {dl.max_abs_d:.4f}")
            lines.append(f"- **Mean |d|:** {dl.mean_abs_d:.4f}")
            lines.append(f"- **Median |d|:** {dl.median_abs_d:.4f}")
            lines.append(
                f"- **Direction:** {dl.n_positive} positive, "
                f"{dl.n_negative} negative, {dl.n_zero} zero"
            )
            if dl.positive_fraction is not None:
                lines.append(f"- **Positive fraction:** {dl.positive_fraction:.3f}")
            lines.append("")
            lines.append("### Threshold Sweep\n")
            lines.append("| Threshold | Cells exceeding |")
            lines.append("|-----------|----------------|")
            for t_str in sorted(dl.threshold_sweep, key=float):
                lines.append(f"| |d| > {t_str} | {dl.threshold_sweep[t_str]} |")
        lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    lines.append(
        "- **GQA non-independence:** Mistral uses 8 KV groups with 32 Q heads. "
        "The 1024 head-level tests are not independent. "
        "Grouped/cluster-aware analysis is a separate follow-up."
    )
    n_inc = provenance.get("n_incorrect", "?")
    n_cor = provenance.get("n_correct", "?")
    lines.append(
        f"- **Small n:** {n_inc} incorrect vs {n_cor} correct. All effect sizes are exploratory."
    )
    lines.append(
        "- **Transform-family dependence:** Raw, diff, and residual modes "
        "are transforms of the same series. Cross-mode recurrence is "
        "robustness evidence, not independent replication."
    )
    lines.append("")

    # Provenance
    lines.append("## Provenance\n")
    pe_cfg = provenance.get("pe_config", {})
    lines.append(f"- **Samples:** `{provenance.get('samples_path', 'unknown')}`")
    lines.append(f"- **Review:** `{provenance.get('review_path', 'unknown')}`")
    lines.append(
        f"- **PE config:** D={pe_cfg.get('D')}, tau={pe_cfg.get('tau')}, "
        f"min_windows_factor={pe_cfg.get('min_windows_factor')}"
    )
    lines.append(
        f"- **Grid:** {provenance.get('num_layers')} layers x {provenance.get('num_heads')} heads"
    )
    lines.append(
        f"- **Samples:** {provenance.get('n_correct')} correct, "
        f"{provenance.get('n_incorrect')} incorrect"
    )
    lines.append("")

    return "\n".join(lines)


# -- Confound controls report (PR #30) --


def _format_asymmetry_section(
    title: str,
    result: AsymmetryNullResult,
) -> list[str]:
    """Format one asymmetry analysis section."""
    lines: list[str] = []
    obs = result.observed
    lines.append(f"### {title}\n")
    lines.append(
        f"- **Voting heads:** {obs.n_negative_heads + obs.n_positive_heads + obs.n_zero_heads}"
    )
    lines.append(
        f"- **Negative:** {obs.n_negative_heads}, **Positive:** {obs.n_positive_heads}, **Zero:** {obs.n_zero_heads}"
    )
    lines.append(
        f"- **Absent:** {obs.n_absent_heads}, **Sparse (<{obs.min_present_combos} combos):** {obs.n_sparse_heads}"
    )
    lines.append(f"- **Signed excess (n_neg - n_pos):** {obs.signed_excess}")
    if obs.negative_fraction is not None:
        lines.append(f"- **Negative fraction:** {obs.negative_fraction:.3f}")
    if obs.mean_head_mean_d is not None:
        lines.append(f"- **Mean head mean-d:** {obs.mean_head_mean_d:.4f}")
        lines.append(f"- **Mean head |mean-d|:** {obs.mean_head_abs_mean_d:.4f}")
    lines.append(f"- **p-value (two-sided, PRIMARY):** {result.p_value_two_sided:.4f}")
    lines.append(
        f"- **p-value (one-sided negative, secondary):** {result.p_value_one_sided_negative:.4f}"
    )
    lines.append(f"- **N permutations:** {result.n_permutations}")
    summary = result.null_signed_excess_summary
    lines.append(
        f"- **Null signed excess:** mean={summary.mean:.1f}, "
        f"std={summary.std:.1f}, "
        f"range=[{summary.min_val:.0f}, {summary.max_val:.0f}]"
    )
    lines.append("")
    return lines


def _format_baseline_deviation(dev: BaselineDeviation) -> list[str]:
    """Format baseline deviation diagnostic."""
    return [
        f"- **Baseline deviation from full cohort:** "
        f"max={dev.max_abs_deviation:.6f}, mean={dev.mean_abs_deviation:.6f} "
        f"({dev.n_positions_compared} positions compared)",
    ]


def format_confound_controls_markdown(
    full_cohort: AsymmetryNullResult,
    matched_result: AsymmetryNullResult | None,
    matched_diag: MatchingDiagnostics | None,
    matched_sensitivity: AsymmetryNullResult | None,
    unanimous_result: AsymmetryNullResult | None,
    unanimous_diag: SelectionDiagnostics | None,
    provenance: dict[str, Any],
    matched_baseline_dev: BaselineDeviation | None = None,
    unanimous_baseline_dev: BaselineDeviation | None = None,
) -> str:
    """Render confound controls report as markdown.

    Markdown order per spec:
    1. Full-cohort asymmetry result
    2. Length-matched diagnostics, then result
    3. Unanimous-only diagnostics, then result
    4. Caveats
    """
    lines: list[str] = []
    lines.append("# PE Confound Controls Report\n")

    # 1. Full-cohort
    lines.append("## Analysis 1: Full-Cohort Signed Asymmetry\n")
    lines.extend(_format_asymmetry_section("Full Cohort", full_cohort))

    # 2. Length-matched
    lines.append("## Analysis 2: Length-Matched\n")
    if matched_diag is not None:
        lines.append("### Matching Diagnostics\n")
        lines.append(
            f"- **Before:** {matched_diag.n_correct_before} correct, {matched_diag.n_incorrect_before} incorrect"
        )
        lines.append(
            f"- **After:** {matched_diag.n_correct_after} correct, {matched_diag.n_incorrect_after} incorrect"
        )
        lines.append(
            f"- **Dropped:** {matched_diag.n_correct_dropped} correct, {matched_diag.n_incorrect_dropped} incorrect"
        )
        lines.append(
            f"- **Mean tokens (before):** correct={matched_diag.mean_tokens_correct_before:.1f}, incorrect={matched_diag.mean_tokens_incorrect_before:.1f}"
        )
        lines.append(
            f"- **Mean tokens (after):** correct={matched_diag.mean_tokens_correct_after:.1f}, incorrect={matched_diag.mean_tokens_incorrect_after:.1f}"
        )
        lines.append(
            f"- **Pair gaps:** max={matched_diag.max_pair_token_gap}, mean={matched_diag.mean_pair_token_gap:.1f}"
        )
        lines.append(f"- **Dropped correct tokens:** {matched_diag.dropped_correct_token_summary}")
        if matched_baseline_dev is not None:
            lines.extend(_format_baseline_deviation(matched_baseline_dev))
        lines.append("")
    if matched_result is not None:
        lines.extend(
            _format_asymmetry_section(
                "Length-Matched (pair-restricted null, PRIMARY)", matched_result
            )
        )
    if matched_sensitivity is not None:
        lines.extend(
            _format_asymmetry_section(
                "Length-Matched (stratified null, sensitivity)", matched_sensitivity
            )
        )

    # 3. Unanimous-only
    lines.append("## Analysis 3: Unanimous-Only\n")
    if unanimous_diag is not None:
        lines.append("### Selection Diagnostics\n")
        lines.append(
            f"- **Before:** {unanimous_diag.n_correct_before} correct, {unanimous_diag.n_incorrect_before} incorrect"
        )
        lines.append(
            f"- **After:** {unanimous_diag.n_correct_after} correct, {unanimous_diag.n_incorrect_after} incorrect"
        )
        lines.append(f"- **Excluded ambiguous:** {unanimous_diag.n_excluded_ambiguous}")
        lines.append(f"- **Excluded non-unanimous:** {unanimous_diag.n_excluded_non_unanimous}")
        if unanimous_baseline_dev is not None:
            lines.extend(_format_baseline_deviation(unanimous_baseline_dev))
        lines.append("")
    if unanimous_result is not None:
        lines.extend(_format_asymmetry_section("Unanimous Only", unanimous_result))

    # 4. Caveats
    lines.append("## Caveats\n")
    lines.append(
        "- **GQA non-independence:** Mistral uses 8 KV groups with 32 Q heads. "
        "The 1024 head-level tests are not independent."
    )
    lines.append(
        "- **Exploratory:** All three analyses are on the same 400-sample data. "
        "No multiple-testing correction applied."
    )
    lines.append(
        "- **Data-discovered direction:** The negative direction was observed "
        "on the same data being tested. Two-sided p-value is primary."
    )
    lines.append(
        "- **Transform-family dependence:** Raw, diff, and residual modes "
        "are transforms of the same series."
    )
    lines.append("")

    # Provenance
    lines.append("## Provenance\n")
    lines.append(f"- **Samples:** `{provenance.get('samples_path', 'unknown')}`")
    lines.append(f"- **Review:** `{provenance.get('review_path', 'unknown')}`")
    pe_cfg = provenance.get("pe_config", {})
    lines.append(
        f"- **PE config:** D={pe_cfg.get('D')}, tau={pe_cfg.get('tau')}, "
        f"min_windows_factor={pe_cfg.get('min_windows_factor')}"
    )
    lines.append(
        f"- **Grid:** {provenance.get('num_layers')} layers x {provenance.get('num_heads')} heads"
    )
    lines.append(
        f"- **Full-cohort samples:** {provenance.get('n_correct')} correct, "
        f"{provenance.get('n_incorrect')} incorrect"
    )
    lines.append("")

    return "\n".join(lines)
