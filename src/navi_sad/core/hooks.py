"""Mock hook manager for SAD capture during inference.

Installs PyTorch forward pre-hooks and post-hooks on attention modules
to capture Q/K/V tensors and compute Spectral Attention Divergence deltas
without modifying the model's output (non-interference invariant).

NOTE: This module uses a mock approach that recomputes Q/K/V from the
module's projection weights. Real family-specific adapters will intercept
post-RoPE tensors directly. See the adapter layer for production use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from navi_sad.core.spectral import (
    expand_kv_heads,
    linear_attention_last_token,
    per_head_cosine_distance,
    softmax_attention_last_token,
)
from navi_sad.core.types import StepRecord


def compute_sad_delta(
    q_last: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    sink_exclude: int = 0,
) -> torch.Tensor:
    """Compute per-head SAD delta between softmax and linear attention.

    All inputs must be fp32. Applies sink exclusion by slicing the first
    N positions from k_prefix and v_prefix before computing attention.

    Args:
        q_last: Query for the newest token. Shape [B, H, 1, D].
        k_prefix: Keys for all prefix positions. Shape [B, H, L, D].
        v_prefix: Values for all prefix positions. Shape [B, H, L, D].
        sink_exclude: Number of initial positions to skip (attention sinks).

    Returns:
        Per-head cosine distance. Shape [H].
    """
    # Apply sink exclusion: skip the first N positions
    if sink_exclude > 0:
        k_prefix = k_prefix[:, :, sink_exclude:, :]
        v_prefix = v_prefix[:, :, sink_exclude:, :]

    # Guard: if prefix is empty after exclusion, return zeros
    if k_prefix.shape[2] == 0:
        num_heads = q_last.shape[1]
        return torch.zeros(num_heads, dtype=q_last.dtype, device=q_last.device)

    out_softmax = softmax_attention_last_token(q_last, k_prefix, v_prefix)
    out_linear = linear_attention_last_token(q_last, k_prefix, v_prefix)
    return per_head_cosine_distance(out_softmax, out_linear)


class HookManager:
    """Manages SAD capture hooks on attention modules.

    Installs a forward pre-hook (to capture Q/K/V) and a forward hook
    (to compute SAD deltas) on each registered module. The hooks are
    observation-only: they must NOT alter the module's output.

    NOTE: This uses the mock approach -- Q/K/V are recomputed from the
    module's projection weights in the pre-hook. Real family-specific
    adapters intercept post-RoPE tensors. This mock approach is suitable
    for testing the hook plumbing and for models without RoPE.
    """

    def __init__(self, sink_exclude: int = 1) -> None:
        self._sink_exclude = sink_exclude
        self._step_idx: int = 0
        self._records: list[StepRecord] = []
        self._handles: list[RemovableHandle] = []

    @property
    def is_installed(self) -> bool:
        """True if hooks are currently installed on at least one module."""
        return len(self._handles) > 0

    def install_on_module(
        self,
        attn_module: nn.Module,
        layer_idx: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> None:
        """Install pre-hook and post-hook on the given attention module.

        Args:
            attn_module: The attention module to instrument.
            layer_idx: Layer index for record attribution.
            num_q_heads: Number of query attention heads.
            num_kv_heads: Number of key/value attention heads (for GQA).
        """
        # --- Pre-hook: capture Q/K/V from module projections ---
        # MOCK ONLY: Real adapters intercept post-RoPE tensors directly.
        # This recomputation approach is valid only for testing and for
        # architectures where the projection output equals the attention input.
        def pre_hook(module: nn.Module, args: tuple) -> None:
            with torch.no_grad():
                hidden_states = args[0]
                B, L, _ = hidden_states.shape
                head_dim = module.head_dim  # type: ignore[union-attr]

                # Use closure variables for head counts, not module.num_heads.
                # For GQA models, K/V projections output num_kv_heads * head_dim,
                # which is smaller than Q's num_q_heads * head_dim.
                q = (
                    module.q_proj(hidden_states)  # type: ignore[union-attr]
                    .view(B, L, num_q_heads, head_dim)
                    .transpose(1, 2)
                )
                k = (
                    module.k_proj(hidden_states)  # type: ignore[union-attr]
                    .view(B, L, num_kv_heads, head_dim)
                    .transpose(1, 2)
                )
                v = (
                    module.v_proj(hidden_states)  # type: ignore[union-attr]
                    .view(B, L, num_kv_heads, head_dim)
                    .transpose(1, 2)
                )

                module._sad_capture = (q, k, v)  # type: ignore[union-attr]

        # --- Post-hook: compute SAD delta from captured tensors ---
        def post_hook(module: nn.Module, args: tuple, output: torch.Tensor) -> None:
            capture = getattr(module, "_sad_capture", None)
            if capture is None:
                return

            q, k, v = capture

            # Detach, clone, upcast to float32
            q = q.detach().clone().float()
            k = k.detach().clone().float()
            v = v.detach().clone().float()

            # Apply GQA expansion if needed
            if num_kv_heads != num_q_heads:
                k = expand_kv_heads(k, num_q_heads)
                v = expand_kv_heads(v, num_q_heads)

            # Take newest-token Q slice: [B, H, 1, D]
            q_last = q[:, :, -1:, :]

            delta = compute_sad_delta(
                q_last, k, v, sink_exclude=self._sink_exclude
            )

            self._records.append(
                StepRecord(
                    step_idx=self._step_idx,
                    layer_idx=layer_idx,
                    per_head_delta=delta.tolist(),
                )
            )

            # Capture hygiene: explicitly delete to prevent lingering attributes
            del module._sad_capture  # type: ignore[union-attr]

        h_pre = attn_module.register_forward_pre_hook(pre_hook)
        h_post = attn_module.register_forward_hook(post_hook)
        self._handles.append(h_pre)
        self._handles.append(h_post)

    def uninstall(self) -> None:
        """Remove all installed hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def step(self) -> None:
        """Increment step_idx. Call once per forward pass, not per hook fire."""
        self._step_idx += 1

    def reset(self) -> None:
        """Zero step_idx and clear all records. Call between samples."""
        self._step_idx = 0
        self._records.clear()

    def get_records(self) -> list[StepRecord]:
        """Return accumulated StepRecords."""
        return list(self._records)
