"""Core data types for navi-SAD instrument."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from navi_sad.core.adapter import MistralAdapter


@dataclass
class CaptureRecord:
    """Ephemeral per-step capture. NOT serialized --- freed after computing deltas."""

    run_id: str
    step_idx: int
    layer_idx: int
    q_last: torch.Tensor  # [1, num_heads, 1, head_dim]
    k_prefix: torch.Tensor  # [1, num_kv_heads, seq_len, head_dim]
    v_prefix: torch.Tensor  # [1, num_kv_heads, seq_len, head_dim]


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
