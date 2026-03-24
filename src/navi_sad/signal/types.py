"""Derived analysis record types."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrdinalResult:
    """Ordinal pattern analysis result."""
    pe: Optional[float]         # None if too few strict-order windows
    tie_rate: float
    pattern_counts: dict[int, int]
    D: int
    tau: int


@dataclass
class DerivedSampleRecord:
    """Re-generable analysis record. Computed from raw records."""
    schema_version: int = 1
    record_type: str = "derived"
    sample_id: str = ""
    source_run_id: str = ""
    per_token_delta: list[float] = field(default_factory=list)
    delta_prime: list[float] = field(default_factory=list)
    delta_double_prime: list[float] = field(default_factory=list)
    delta_triple_prime: list[float] = field(default_factory=list)
    ordinal: Optional[OrdinalResult] = None
    summary: dict = field(default_factory=dict)
    aggregation_method: str = "uniform_mean"
    analysis_version: str = ""
