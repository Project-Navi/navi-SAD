"""Core data types for navi-SAD instrument."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navi_sad.core.adapter import MistralAdapter


@dataclass
class StepRecord:
    """Computed per-step result. Serialized in raw records."""

    step_idx: int
    layer_idx: int
    per_head_delta: list[float]  # [num_heads] cosine distances


@dataclass
class RawSampleRecord:
    """Immutable raw inference record. Written once, never modified."""

    schema_version: int = 1
    record_type: str = "raw"
    sample_id: str = ""
    model: str = ""
    benchmark: str = ""
    prompt: str = ""
    generation: str = ""
    label: str = ""  # "correct" | "incorrect" | "ambiguous"
    label_source: str = ""
    scorer_version: str = ""
    num_tokens_generated: int = 0
    layers_hooked: list[int] = field(default_factory=list)
    capture_tier: str = ""
    per_step: list[StepRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelFamilyConfig:
    """Registry entry for a model family."""

    architecture: str
    attn_module_path: str  # e.g. "model.layers.{}.self_attn"
    capture_tier: str  # "A", "B", or "C"
    num_kv_heads_attr: str  # config attribute name
    num_q_heads_attr: str
    head_dim_attr: str
    gqa_expansion: bool
    notes: str = ""
    adapter_factory: Callable[[], MistralAdapter] | None = None


@dataclass
class ParityConfig:
    """Configuration for Gate 1 parity validation mode."""

    enabled: bool = True
    include_pre_oproj: bool = True


@dataclass
class ParityRecord:
    """Per-layer parity check result for Gate 1.

    Gate metrics (cosine_similarity, relative_l2_error) have frozen
    thresholds applied. Diagnostics (max_absolute_error, pre_oproj_cosine)
    are reported but not gated.

    pre_oproj_cosine is None when ParityConfig.include_pre_oproj is False,
    not when computation failed.
    """

    layer_idx: int
    step_idx: int
    # Gate metrics
    cosine_similarity: float
    relative_l2_error: float
    # Diagnostics
    max_absolute_error: float
    pre_oproj_cosine: float | None
