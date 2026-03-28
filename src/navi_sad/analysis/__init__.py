"""PE recurrence null analysis instrument."""

from navi_sad.analysis.eligibility import build_eligibility_table
from navi_sad.analysis.permutation import run_permutation_null
from navi_sad.analysis.recurrence import build_pe_lookup, compute_recurrence
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
    "EligibilityCell",
    "EligibilityTable",
    "PermutationNullConfig",
    "PermutationNullResult",
    "RecurrenceNullReport",
    "RecurrenceProfile",
    "RecurrenceStatistic",
    "build_eligibility_table",
    "build_pe_lookup",
    "compute_recurrence",
    "run_permutation_null",
]
