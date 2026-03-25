"""Tests for MistralAdapter forward-replacement capture."""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
)

from navi_sad.core.adapter import MistralAdapter


def _make_small_attn(
    hidden_size: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
) -> tuple[MistralAttention, MistralConfig]:
    """Create a tiny MistralAttention for CPU testing."""
    config = MistralConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=128,
        _attn_implementation="eager",
    )
    attn = MistralAttention(config, layer_idx=0)
    attn.eval()
    return attn, config


_CPU = torch.device("cpu")


def _make_position_embeddings(
    config: MistralConfig,
    seq_len: int,
    device: torch.device = _CPU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate cos/sin position embeddings deterministically."""
    rope = MistralRotaryEmbedding(config=config).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    x = torch.zeros(1, seq_len, config.hidden_size, device=device)
    cos, sin = rope(x, position_ids)
    return cos, sin


class TestAdapterLifecycle:
    def test_install_stores_original_forward(self) -> None:
        attn, _ = _make_small_attn()
        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)
        assert hasattr(attn, "_sad_original_forward")
        # Bound methods are new objects each access; check callable is stored
        assert callable(attn._sad_original_forward)

    def test_uninstall_restores_original_forward(self) -> None:
        attn, _ = _make_small_attn()
        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)
        adapter.uninstall(attn)
        assert not hasattr(attn, "_sad_original_forward")

    def test_double_install_raises(self) -> None:
        attn, _ = _make_small_attn()
        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)
        with pytest.raises(RuntimeError, match="already patched"):
            adapter.install(attn, capture_fn=lambda q, k, v: None)

    def test_version_guard_present(self) -> None:
        """Adapter must check transformers version at install time."""
        attn, _ = _make_small_attn()
        adapter = MistralAdapter()
        # Should not raise for the pinned version
        adapter.install(attn, capture_fn=lambda q, k, v: None)
        adapter.uninstall(attn)

    def test_incompatible_version_raises(self) -> None:
        """Version guard rejects incompatible transformers versions."""
        from navi_sad.core import adapter as adapter_mod

        attn, _ = _make_small_attn()
        ad = MistralAdapter()
        original_min = adapter_mod._COMPAT_MIN
        try:
            # Temporarily set a range that excludes the current version
            adapter_mod._COMPAT_MIN = "99.0.0"
            with pytest.raises(RuntimeError, match="requires transformers"):
                ad.install(attn, capture_fn=lambda q, k, v: None)
        finally:
            adapter_mod._COMPAT_MIN = original_min

    def test_unparseable_version_raises(self) -> None:
        """Version guard rejects unparseable version strings."""
        import unittest.mock

        from navi_sad.core.adapter import _check_transformers_version

        with unittest.mock.patch("transformers.__version__", "not.a.version"):
            with pytest.raises(RuntimeError, match="Cannot parse"):
                _check_transformers_version()

    def test_non_eager_raises_on_forward(self) -> None:
        """Adapter raises RuntimeError on non-eager attention implementation."""
        attn, config = _make_small_attn()
        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)

        # Patch the config to simulate non-eager after install
        attn.config._attn_implementation = "sdpa"

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with pytest.raises(RuntimeError, match="attn_implementation='eager'"):
            with torch.no_grad():
                attn(hidden_states, pos_emb, None)

    def test_signature_parity(self) -> None:
        """Patched forward must accept the same parameter names as original."""
        attn, _ = _make_small_attn()
        original_sig = inspect.signature(attn.forward)

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)
        patched_sig = inspect.signature(attn.forward)

        orig_params = set(original_sig.parameters.keys())
        patched_params = set(patched_sig.parameters.keys())

        # Patched is a closure (no 'self'), original is a bound method
        orig_params.discard("self")
        patched_params.discard("self")
        assert orig_params == patched_params, (
            f"Signature mismatch.\nOriginal: {orig_params}\nPatched:  {patched_params}"
        )


class TestAdapterCapture:
    def test_capture_fn_called_with_post_rope_shapes(self) -> None:
        """Capture callback receives [B, H, L, D] tensors."""
        attn, config = _make_small_attn(hidden_size=64, num_heads=4, num_kv_heads=2)
        captured: dict[str, torch.Tensor] = {}

        def capture_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
            captured["q"] = q.detach().clone()
            captured["k"] = k.detach().clone()
            captured["v"] = v.detach().clone()

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=capture_fn)

        B, L, D = 1, 8, 64
        head_dim = D // config.num_attention_heads

        hidden_states = torch.randn(B, L, D)
        pos_emb = _make_position_embeddings(config, L)

        with torch.no_grad():
            attn(hidden_states, pos_emb, None)

        assert "q" in captured
        assert captured["q"].shape == (B, config.num_attention_heads, L, head_dim)
        assert captured["k"].shape == (B, config.num_key_value_heads, L, head_dim)
        assert captured["v"].shape == (B, config.num_key_value_heads, L, head_dim)

    def test_capture_not_called_after_uninstall(self) -> None:
        attn, config = _make_small_attn()
        call_count = 0

        def capture_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
            nonlocal call_count
            call_count += 1

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=capture_fn)
        adapter.uninstall(attn)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)

        assert call_count == 0


class TestAdapterNonInterference:
    def test_patched_output_exact_match(self) -> None:
        """Patched forward produces bit-identical output to original.

        If this fails, the adapter is wrong. Not "close enough."
        """
        attn, config = _make_small_attn()
        torch.manual_seed(42)

        B, L, D = 1, 8, 64
        hidden_states = torch.randn(B, L, D)
        pos_emb = _make_position_embeddings(config, L)

        with torch.no_grad():
            out_original, _ = attn(hidden_states, pos_emb, None)

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)

        with torch.no_grad():
            out_patched, _ = attn(hidden_states, pos_emb, None)

        torch.testing.assert_close(out_patched, out_original, rtol=0, atol=0)

    def test_patched_output_exact_match_gqa(self) -> None:
        """Non-interference holds under GQA (8 Q heads, 2 KV heads)."""
        attn, config = _make_small_attn(hidden_size=128, num_heads=8, num_kv_heads=2)
        torch.manual_seed(99)

        B, L, D = 1, 12, 128
        hidden_states = torch.randn(B, L, D)
        pos_emb = _make_position_embeddings(config, L)

        with torch.no_grad():
            out_original, _ = attn(hidden_states, pos_emb, None)

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)

        with torch.no_grad():
            out_patched, _ = attn(hidden_states, pos_emb, None)

        torch.testing.assert_close(out_patched, out_original, rtol=0, atol=0)

    def test_return_arity_matches_original(self) -> None:
        """Patched forward returns same tuple structure as original."""
        attn, config = _make_small_attn()
        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)

        with torch.no_grad():
            result_original = attn(hidden_states, pos_emb, None)

        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)

        with torch.no_grad():
            result_patched = attn(hidden_states, pos_emb, None)

        assert type(result_original) is type(result_patched)
        assert len(result_original) == len(result_patched)


class TestAdapterParityCallback:
    def test_parity_fn_receives_pre_oproj_output(self) -> None:
        """Parity callback receives pre-o_proj newest-token tensor."""
        attn, config = _make_small_attn()
        parity_data: dict = {}

        def parity_fn(**kwargs) -> None:
            parity_data.update(kwargs)

        adapter = MistralAdapter()
        adapter.install(
            attn,
            capture_fn=lambda q, k, v: None,
            parity_fn=parity_fn,
        )

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)

        assert "pre_oproj_output" in parity_data
        assert parity_data["pre_oproj_output"] is not None
        # Shape: [B, 1, H, D] -- newest token along seq dim, pre-reshape
        head_dim = 64 // config.num_attention_heads
        assert parity_data["pre_oproj_output"].shape == (
            1,
            1,
            config.num_attention_heads,
            head_dim,
        )

    def test_no_parity_fn_means_no_pre_oproj_capture(self) -> None:
        """Without parity_fn, no pre-o_proj capture overhead."""
        attn, config = _make_small_attn()
        adapter = MistralAdapter()
        adapter.install(attn, capture_fn=lambda q, k, v: None)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            out, _ = attn(hidden_states, pos_emb, None)

        # Should still produce valid output (non-interference)
        assert out.shape == (1, 8, 64)
