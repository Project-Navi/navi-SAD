"""Report rendering for PE recurrence null analysis.

Builds provenance dicts and renders markdown reports.
Tested module — not in the CLI script.
"""

from __future__ import annotations

from typing import Any

from navi_sad.analysis.loader import AnalysisInput
from navi_sad.analysis.types import RecurrenceNullReport
from navi_sad.signal.pe_features import PEConfig


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
