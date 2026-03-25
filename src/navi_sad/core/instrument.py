"""InstrumentManager -- orchestrates real model instrumentation.

Coordinates MistralAdapter instances across layers, manages step
accounting, and computes SAD deltas from captured post-RoPE tensors.
This is the real-model counterpart to the mock HookManager in hooks.py.

Parity mode (Gate 1) is added in Phase C4.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from navi_sad.core.adapter import MistralAdapter
from navi_sad.core.hooks import compute_sad_delta
from navi_sad.core.spectral import expand_kv_heads
from navi_sad.core.types import StepRecord

logger = logging.getLogger(__name__)


class InstrumentManager:
    """Manages SAD instrumentation across model layers.

    Usage::

        mgr = InstrumentManager(sink_exclude=1)
        for layer_idx, attn in enumerate(attention_layers):
            mgr.install_layer(attn, layer_idx, num_q_heads, num_kv_heads)

        for token_step in generation_loop:
            model.forward(...)   # adapters capture during forward
            mgr.step()           # increment step_idx

        records = mgr.get_records()
        mgr.reset()             # between samples
    """

    def __init__(self, sink_exclude: int = 1) -> None:
        self._sink_exclude = sink_exclude
        self._step_idx: int = 0
        self._records: list[StepRecord] = []
        self._adapter = MistralAdapter()
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

        self._adapter.install(attn_module, capture_fn=capture_fn)
        self._installed_modules.append(attn_module)

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

    def get_records(self) -> list[StepRecord]:
        """Return accumulated StepRecords."""
        return list(self._records)

    @property
    def is_installed(self) -> bool:
        return len(self._installed_modules) > 0
