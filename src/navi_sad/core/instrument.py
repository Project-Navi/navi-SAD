"""InstrumentManager -- orchestrates real model instrumentation.

Coordinates MistralAdapter instances across layers, manages step
accounting, and computes SAD deltas from captured post-RoPE tensors.
This is the real-model counterpart to the mock HookManager in hooks.py.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from navi_sad.core.hooks import compute_sad_delta
from navi_sad.core.spectral import expand_kv_heads, softmax_attention_last_token
from navi_sad.core.types import ModelFamilyConfig, ParityConfig, ParityRecord, StepRecord

if TYPE_CHECKING:
    from navi_sad.core.adapter import MistralAdapter

logger = logging.getLogger(__name__)


class InstrumentManager:
    """Manages SAD instrumentation across model layers.

    Usage::

        mgr = InstrumentManager(family_config, sink_exclude=1)
        for layer_idx, attn in enumerate(attention_layers):
            mgr.install_layer(attn, layer_idx, num_q_heads, num_kv_heads)

        for token_step in generation_loop:
            model.forward(...)   # adapters capture during forward
            mgr.step()           # increment step_idx

        records = mgr.get_records()
        mgr.reset()             # between samples
    """

    def __init__(
        self,
        family_config: ModelFamilyConfig,
        sink_exclude: int = 1,
        parity: ParityConfig | None = None,
    ) -> None:
        if family_config.adapter_factory is None:
            raise ValueError(
                f"Family config for '{family_config.architecture}' has no adapter_factory. "
                f"Cannot instrument without an adapter."
            )
        self._sink_exclude = sink_exclude
        self._parity = parity
        self._step_idx: int = 0
        self._records: list[StepRecord] = []
        self._parity_records: list[ParityRecord] = []
        self._adapter: MistralAdapter = family_config.adapter_factory()
        self._installed_modules: list[nn.Module] = []

    def install_layer(
        self,
        attn_module: nn.Module,
        layer_idx: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        """Install capture adapter on an attention module."""

        def capture_fn(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> None:
            with torch.no_grad():
                q_fp32 = q.detach().clone().float()
                k_fp32 = k.detach().clone().float()
                v_fp32 = v.detach().clone().float()

                if num_kv_heads != num_q_heads:
                    k_fp32 = expand_kv_heads(k_fp32, num_q_heads)
                    v_fp32 = expand_kv_heads(v_fp32, num_q_heads)

                q_last = q_fp32[:, :, -1:, :]

                delta = compute_sad_delta(q_last, k_fp32, v_fp32, sink_exclude=self._sink_exclude)

                self._records.append(
                    StepRecord(
                        step_idx=self._step_idx,
                        layer_idx=layer_idx,
                        per_head_delta=delta.tolist(),
                    )
                )

        parity_fn = None
        if self._parity is not None and self._parity.enabled:
            parity_fn = self._make_parity_fn(layer_idx, num_q_heads, num_kv_heads)

        self._adapter.install(attn_module, capture_fn=capture_fn, parity_fn=parity_fn)
        self._installed_modules.append(attn_module)

    def _make_parity_fn(
        self,
        layer_idx: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> Callable[..., None]:
        """Build parity callback for Gate 1 validation."""

        def parity_fn(
            *,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            native_output: torch.Tensor,
            o_proj: nn.Module,
            pre_oproj_output: torch.Tensor | None = None,
        ) -> None:
            with torch.no_grad():
                # Single-sequence only (no padding support)
                if native_output.shape[0] != 1:
                    raise ValueError(f"Parity requires B=1, got {native_output.shape[0]}")

                # Determine native dtype before upcasting
                native_dtype = native_output.dtype

                # fp32 recomputation from captured post-RoPE Q/K/V
                q_fp32 = query_states.detach().clone().float()
                k_fp32 = key_states.detach().clone().float()
                v_fp32 = value_states.detach().clone().float()

                # GQA expansion (required: callback receives pre-expansion K/V)
                if num_kv_heads != num_q_heads:
                    k_fp32 = expand_kv_heads(k_fp32, num_q_heads)
                    v_fp32 = expand_kv_heads(v_fp32, num_q_heads)

                # fp32 softmax attention for newest token
                q_last = q_fp32[:, :, -1:, :]  # [B, H, 1, D]
                softmax_out = softmax_attention_last_token(q_last, k_fp32, v_fp32)
                # softmax_out: [B, H, 1, D]

                # Head merge: transpose THEN reshape (order matters!)
                # [B, H, 1, D] -> transpose(1,2) -> [B, 1, H, D] -> reshape -> [B, 1, H*D]
                batch = softmax_out.shape[0]
                merged = softmax_out.transpose(1, 2).reshape(batch, 1, -1)  # [B, 1, H*D]

                # Downcast to native dtype, pass through o_proj
                recomputed = o_proj(merged.to(native_dtype))  # [B, 1, hidden_size]

                # Native output: SLICE newest token (full [B, L, hidden_size] arrives)
                native_slice = native_output[:, -1:, :].detach()  # [B, 1, hidden_size]

                # Upcast both to fp32 for metric computation
                recomputed_f32 = recomputed.float()
                native_f32 = native_slice.float()

                # Gate metric: cosine similarity
                cos_sim = F.cosine_similarity(
                    recomputed_f32.flatten(), native_f32.flatten(), dim=0
                ).item()

                # Gate metric: relative L2
                # Formula: ||recomputed - native||_2 / (||native||_2 + eps)
                diff = recomputed_f32 - native_f32
                eps = 1e-12
                rel_l2 = diff.norm().item() / (native_f32.norm().item() + eps)

                # Diagnostic: max absolute error
                max_abs = diff.abs().max().item()

                # Diagnostic: pre-o_proj cosine (optional)
                pre_oproj_cosine: float | None = None
                if (
                    pre_oproj_output is not None
                    and self._parity is not None
                    and self._parity.include_pre_oproj
                ):
                    # pre_oproj_output: [B, 1, H, D] from insertion 3
                    # Transpose to [B, H, 1, D] to match softmax_out shape
                    native_pre = pre_oproj_output.transpose(1, 2).float()
                    pre_oproj_cosine = F.cosine_similarity(
                        softmax_out.flatten(), native_pre.flatten(), dim=0
                    ).item()

                self._parity_records.append(
                    ParityRecord(
                        layer_idx=layer_idx,
                        step_idx=self._step_idx,
                        cosine_similarity=cos_sim,
                        relative_l2_error=rel_l2,
                        max_absolute_error=max_abs,
                        pre_oproj_cosine=pre_oproj_cosine,
                    )
                )

        return parity_fn

    def make_step_callback(self):  # type: ignore[return]
        """Return a LogitsProcessor that increments step_idx.

        Use with model.generate() via LogitsProcessorList::

            logits_processor=LogitsProcessorList([mgr.make_step_callback()])
        """
        from transformers import LogitsProcessor

        mgr = self

        class StepCounter(LogitsProcessor):
            def __call__(
                self,
                input_ids: torch.LongTensor,
                scores: torch.FloatTensor,
            ) -> torch.FloatTensor:
                mgr.step()
                return scores

        return StepCounter()

    def uninstall(self) -> None:
        """Remove adapters from all installed modules."""
        for module in self._installed_modules:
            self._adapter.uninstall(module)
        self._installed_modules.clear()

    def step(self) -> None:
        """Increment step_idx. Call once per forward pass."""
        self._step_idx += 1

    def reset(self) -> None:
        """Zero step_idx and clear records. Call between samples."""
        self._step_idx = 0
        self._records.clear()
        self._parity_records.clear()

    def get_records(self) -> list[StepRecord]:
        """Return accumulated StepRecords."""
        return list(self._records)

    def get_parity_records(self) -> list[ParityRecord]:
        """Return accumulated ParityRecords (Gate 1 mode only)."""
        return list(self._parity_records)

    @property
    def is_installed(self) -> bool:
        return len(self._installed_modules) > 0
