"""Tests for report rendering.

Proves: markdown section ordering, provenance presence,
required headings, table formatting.
"""

from __future__ import annotations

from navi_sad.analysis.loader import AnalysisInput
from navi_sad.analysis.report import build_provenance, format_markdown
from navi_sad.analysis.types import (
    EligibilityCell,
    EligibilityTable,
    PermutationNullConfig,
    PermutationNullResult,
    RecurrenceNullReport,
    RecurrenceProfile,
    RecurrenceStatistic,
)
from navi_sad.signal.pe_features import PEConfig


def _make_report() -> RecurrenceNullReport:
    return RecurrenceNullReport(
        config=PermutationNullConfig(n_permutations=100),
        eligibility=EligibilityTable(
            cells=[EligibilityCell("raw", "full", 5, 3, 5, 3, 5, 3)],
            n_correct=5,
            n_incorrect=3,
        ),
        observed=RecurrenceStatistic(0.5, 3, 10, 100, {(0, 0): 3}),
        observed_profile=RecurrenceProfile({1: 50, 3: 10, 7: 2, 12: 0}),
        null_at_min_combos=PermutationNullResult(
            observed=10,
            null_counts=[5, 6, 7],
            p_value=0.25,
            expected_under_null=6.0,
            null_mean=6.0,
            null_std=1.0,
            null_min=5,
            null_max=7,
            null_percentiles={50: 6},
        ),
        null_at_seven=PermutationNullResult(
            observed=2,
            null_counts=[0, 1, 0],
            p_value=0.5,
            expected_under_null=0.33,
            null_mean=0.33,
            null_std=0.47,
            null_min=0,
            null_max=1,
            null_percentiles={50: 0},
        ),
        bin_boundaries=[100],
        bin_counts={"0": {"correct": 3, "incorrect": 1}},
    )


class TestBuildProvenance:
    def test_includes_pe_config(self) -> None:
        data = AnalysisInput(
            labels={1: "correct"},
            token_counts={1: 100},
            per_step_data={1: []},
            n_correct=1,
            n_incorrect=0,
            samples_path="/path/samples.json",
            review_path="/path/review.json",
        )
        prov = build_provenance(data, PEConfig(), num_layers=32, num_heads=32)
        assert "pe_config" in prov
        assert prov["pe_config"]["D"] == 3
        assert prov["pe_config"]["tau"] == 1

    def test_includes_artifact_paths(self) -> None:
        data = AnalysisInput(
            labels={1: "correct"},
            token_counts={1: 100},
            per_step_data={1: []},
            n_correct=1,
            n_incorrect=0,
            samples_path="/data/samples.json",
            review_path="/data/review.json",
        )
        prov = build_provenance(data, PEConfig(), num_layers=32, num_heads=32)
        assert prov["samples_path"] == "/data/samples.json"
        assert prov["review_path"] == "/data/review.json"


class TestFormatMarkdown:
    def test_eligibility_appears_first(self) -> None:
        """Eligibility section must appear before observed recurrence."""
        report = _make_report()
        prov = {"pe_config": {"D": 3, "tau": 1, "min_windows_factor": 2}}
        md = format_markdown(report, prov)
        elig_pos = md.index("## Eligibility")
        obs_pos = md.index("## Observed Recurrence")
        assert elig_pos < obs_pos

    def test_provenance_section_present(self) -> None:
        report = _make_report()
        prov = {
            "samples_path": "/path/samples.json",
            "review_path": "/path/review.json",
            "pe_config": {"D": 3, "tau": 1, "min_windows_factor": 2},
            "num_layers": 32,
            "num_heads": 32,
            "n_correct": 5,
            "n_incorrect": 3,
        }
        md = format_markdown(report, prov)
        assert "## Provenance" in md
        assert "/path/samples.json" in md
        assert "D=3" in md

    def test_caveats_present(self) -> None:
        report = _make_report()
        prov = {"pe_config": {}}
        md = format_markdown(report, prov)
        assert "GQA non-independence" in md
        assert "Small n" in md
        assert "Transform-family dependence" in md

    def test_required_headings(self) -> None:
        report = _make_report()
        prov = {"pe_config": {}}
        md = format_markdown(report, prov)
        for heading in [
            "Eligibility",
            "Observed Recurrence",
            "Recurrence Profile",
            "Permutation Null Test",
            "Caveats",
            "Provenance",
        ]:
            assert heading in md

    def test_table_rows_have_trailing_pipe(self) -> None:
        """All table rows must end with | for valid markdown."""
        report = _make_report()
        prov = {"pe_config": {}}
        md = format_markdown(report, prov)
        for line in md.split("\n"):
            if line.startswith("|") and not line.startswith("|--"):
                assert line.rstrip().endswith("|"), f"Missing trailing pipe: {line}"
