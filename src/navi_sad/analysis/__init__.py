"""PE recurrence null analysis instrument."""

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import AnalysisInput, load_and_validate
from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.prep import PEBundle, SeriesData, compute_pe_bundle, prepare_series_data
from navi_sad.analysis.recurrence import (
    build_pe_lookup,
    compute_d_matrix,
    compute_recurrence,
    recurrence_from_d_matrix,
    summarize_d_matrix,
    validate_combo_set,
)
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

__all__ = [
    "AnalysisInput",
    "EligibilityCell",
    "EligibilityTable",
    "PEBundle",
    "PermutationNullConfig",
    "PermutationNullResult",
    "RecurrenceNullReport",
    "RecurrenceProfile",
    "RecurrenceStatistic",
    "SeriesData",
    "build_eligibility_table",
    "build_pe_lookup",
    "build_provenance",
    "compute_d_matrix",
    "compute_pe_bundle",
    "compute_recurrence",
    "format_markdown",
    "load_and_validate",
    "prepare_series_data",
    "recurrence_from_d_matrix",
    "run_permutation_null",
    "summarize_d_matrix",
    "validate_combo_set",
]
