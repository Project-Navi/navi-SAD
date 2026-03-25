"""Mistral family adapter for Tier A post-RoPE Q/K/V capture.

Replaces MistralAttention.forward with a verbatim copy of the upstream
source from transformers 4.57.x, adding only a capture callback after
apply_rotary_pos_emb and an optional parity callback after o_proj.

ADAPTER DISCIPLINE:
- The patched forward is a VERBATIM COPY of upstream, not a reimplementation.
- Only two insertion points: capture_fn after RoPE, parity_fn after o_proj.
- No refactoring, no kwarg narrowing, no "cleanup" of the upstream code.
- Runtime version guard rejects incompatible transformers versions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Compatible transformers version range for this adapter.
# The patched forward is copied from this exact version range.
_COMPAT_MIN = (4, 57, 0)
_COMPAT_MAX = (4, 58, 0)  # exclusive


def _check_transformers_version() -> None:
    """Raise if transformers version is outside compatible range."""
    import transformers

    version_str = transformers.__version__
    parts = version_str.split(".")
    try:
        version_tuple = tuple(int(p) for p in parts[:3])
    except ValueError as exc:
        raise RuntimeError(f"Cannot parse transformers version: {version_str}") from exc
    if not (_COMPAT_MIN <= version_tuple < _COMPAT_MAX):
        raise RuntimeError(
            f"MistralAdapter requires transformers "
            f"{'.'.join(str(x) for x in _COMPAT_MIN)} "
            f"to <{'.'.join(str(x) for x in _COMPAT_MAX)}, "
            f"got {version_str}. The patched forward is copied from "
            f"this version range and may not be compatible."
        )


class MistralAdapter:
    """Tier A adapter for MistralForCausalLM attention modules.

    Captures post-RoPE Q/K/V by replacing the attention module's forward
    method with a verbatim copy that inserts capture callbacks at two
    marked insertion points.
    """

    def install(
        self,
        attn_module: nn.Module,
        capture_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
        parity_fn: Callable[..., None] | None = None,
    ) -> None:
        """Patch forward to capture post-RoPE Q/K/V.

        Args:
            attn_module: A MistralAttention module.
            capture_fn: Called with (query_states, key_states, value_states)
                after RoPE application. Tensors are live (not cloned) --
                the callback is responsible for detach/clone if needed.
            parity_fn: Optional. Called with (query_states, key_states,
                value_states, native_output, o_proj) for Gate 1 parity.
        """
        _check_transformers_version()

        if hasattr(attn_module, "_sad_original_forward"):
            raise RuntimeError(
                f"Module {attn_module} is already patched. Call uninstall() before re-installing."
            )

        attn_module._sad_original_forward = attn_module.forward  # type: ignore[attr-defined, assignment]
        attn_module.forward = self._make_capturing_forward(  # type: ignore[assignment]
            attn_module, capture_fn, parity_fn
        )

    def uninstall(self, attn_module: nn.Module) -> None:
        """Restore original forward method."""
        original = getattr(attn_module, "_sad_original_forward", None)
        if original is None:
            return
        attn_module.forward = original  # type: ignore[method-assign]
        del attn_module._sad_original_forward  # type: ignore[attr-defined]

    @staticmethod
    def _make_capturing_forward(
        module: nn.Module,
        capture_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
        parity_fn: Callable[..., None] | None = None,
    ) -> Callable:
        """Build patched forward: verbatim upstream copy + capture insertions.

        IMPORTANT: This is a VERBATIM COPY of MistralAttention.forward
        from transformers 4.57.x with two marked insertion points.
        Do not refactor. Do not "clean up." Any deviation from upstream
        is a potential non-interference violation.
        """
        from transformers.models.mistral.modeling_mistral import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        # --- BEGIN: verbatim upstream copy (transformers 4.57.x) ---
        # Source: MistralAttention.forward
        # Only changes: 'self' -> 'module' (closure), two SAD insertions
        def forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            cache_position: torch.LongTensor | None = None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)  # type: ignore[union-attr]

            query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # type: ignore[union-attr, operator]
            key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # type: ignore[union-attr, operator]
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # type: ignore[union-attr, operator]

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === SAD INSERTION 1: Tier A capture post-RoPE Q/K/V ===
            capture_fn(query_states, key_states, value_states)
            # === END INSERTION 1 ===

            if past_key_values is not None:
                # sin and cos are specific to RoPE models;
                # cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_values.update(
                    key_states,
                    value_states,
                    module.layer_idx,
                    cache_kwargs,  # type: ignore[union-attr]
                )

            attention_interface: Callable = eager_attention_forward
            if module.config._attn_implementation != "eager":  # type: ignore[union-attr]
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    module.config._attn_implementation  # type: ignore[union-attr]
                ]

            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,  # type: ignore[union-attr]
                scaling=module.scaling,  # type: ignore[union-attr]
                sliding_window=getattr(module.config, "sliding_window", None),  # type: ignore[union-attr]
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = module.o_proj(attn_output)  # type: ignore[union-attr, operator]

            # === SAD INSERTION 2: parity capture (Gate 1 only) ===
            if parity_fn is not None:
                assert past_key_values is None, (
                    "Parity mode requires use_cache=False. Cache-on parity is not validated."
                )
                parity_fn(
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    native_output=attn_output,
                    o_proj=module.o_proj,  # type: ignore[union-attr]
                )
            # === END INSERTION 2 ===

            return attn_output, attn_weights

        # --- END: verbatim upstream copy ---

        return forward
