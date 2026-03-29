"""PE analysis instrument — recurrence, asymmetry, confound controls."""

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import AnalysisInput, load_and_validate, load_reviewer_votes
from navi_sad.analysis.matching import match_by_token_count
from navi_sad.analysis.permutation import (
    run_asymmetry_null,
    run_paired_asymmetry_null,
    run_permutation_null,
)
from navi_sad.analysis.prep import (
    PEBundle,
    SeriesData,
    compute_baseline_deviation,
    compute_pe_bundle,
    prepare_series_data,
    prepare_series_data_from_subset,
)
from navi_sad.analysis.recurrence import (
    build_pe_lookup,
    compute_d_matrix,
    compute_head_asymmetry,
    compute_recurrence,
    recurrence_from_d_matrix,
    summarize_d_matrix,
    validate_combo_set,
)
from navi_sad.analysis.report import (
    build_provenance,
    format_confound_controls_markdown,
    format_markdown,
)
from navi_sad.analysis.selection import select_unanimous
from navi_sad.analysis.types import (
    CANONICAL_LABELS,
    AsymmetryNullResult,
    AsymmetryStatistic,
    BaselineDeviation,
    DLandscape,
    EligibilityCell,
    EligibilityTable,
    MatchingDiagnostics,
    NullDistributionSummary,
    PermutationNullConfig,
    PermutationNullResult,
    RecurrenceNullReport,
    RecurrenceProfile,
    RecurrenceStatistic,
    SelectionDiagnostics,
    SubsetSpec,
)

__all__ = [
    "CANONICAL_LABELS",
    "AnalysisInput",
    "AsymmetryNullResult",
    "AsymmetryStatistic",
    "BaselineDeviation",
    "DLandscape",
    "EligibilityCell",
    "EligibilityTable",
    "MatchingDiagnostics",
    "NullDistributionSummary",
    "PEBundle",
    "PermutationNullConfig",
    "PermutationNullResult",
    "RecurrenceNullReport",
    "RecurrenceProfile",
    "RecurrenceStatistic",
    "SelectionDiagnostics",
    "SeriesData",
    "SubsetSpec",
    "build_eligibility_table",
    "build_pe_lookup",
    "build_provenance",
    "compute_baseline_deviation",
    "compute_d_matrix",
    "compute_head_asymmetry",
    "compute_pe_bundle",
    "compute_recurrence",
    "format_confound_controls_markdown",
    "format_markdown",
    "load_and_validate",
    "load_reviewer_votes",
    "match_by_token_count",
    "prepare_series_data",
    "prepare_series_data_from_subset",
    "recurrence_from_d_matrix",
    "run_asymmetry_null",
    "run_paired_asymmetry_null",
    "run_permutation_null",
    "select_unanimous",
    "summarize_d_matrix",
    "validate_combo_set",
]
