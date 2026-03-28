"""PE recurrence null analysis instrument."""

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.loader import AnalysisInput, load_and_validate
from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.recurrence import (
    build_pe_lookup,
    compute_recurrence,
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
    "PermutationNullConfig",
    "PermutationNullResult",
    "RecurrenceNullReport",
    "RecurrenceProfile",
    "RecurrenceStatistic",
    "build_eligibility_table",
    "build_pe_lookup",
    "build_provenance",
    "compute_recurrence",
    "format_markdown",
    "load_and_validate",
    "run_permutation_null",
    "validate_combo_set",
]
