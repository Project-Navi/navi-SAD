"""Tests for InstrumentManager -- real model orchestration."""

from __future__ import annotations

import torch
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
)

from navi_sad.core.instrument import InstrumentManager

_CPU = torch.device("cpu")


def _make_small_attn(
    hidden_size: int = 64,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    layer_idx: int = 0,
) -> tuple[MistralAttention, MistralConfig]:
    config = MistralConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=128,
        _attn_implementation="eager",
    )
    attn = MistralAttention(config, layer_idx=layer_idx)
    attn.eval()
    return attn, config


def _make_position_embeddings(
    config: MistralConfig,
    seq_len: int,
    device: torch.device = _CPU,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = MistralRotaryEmbedding(config=config).to(device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    x = torch.zeros(1, seq_len, config.hidden_size, device=device)
    cos, sin = rope(x, position_ids)
    return cos, sin


class TestInstrumentManager:
    def test_single_layer_produces_records(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager(sink_exclude=1)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)

        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        records = mgr.get_records()
        assert len(records) == 1
        assert records[0].step_idx == 0
        assert records[0].layer_idx == 0
        assert len(records[0].per_head_delta) == 4

    def test_multi_step_increments_correctly(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager(sink_exclude=0)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        for step in range(3):
            L = 4 + step
            hidden_states = torch.randn(1, L, 64)
            pos_emb = _make_position_embeddings(config, L)
            with torch.no_grad():
                attn(hidden_states, pos_emb, None)
            mgr.step()

        records = mgr.get_records()
        assert len(records) == 3
        assert [r.step_idx for r in records] == [0, 1, 2]

    def test_reset_clears_state(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager()
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        assert len(mgr.get_records()) == 1
        mgr.reset()
        assert len(mgr.get_records()) == 0

    def test_uninstall_stops_capture(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager()
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)
        mgr.uninstall()

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)

        assert len(mgr.get_records()) == 0

    def test_per_head_delta_in_valid_range(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager(sink_exclude=0)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 10, 64)
        pos_emb = _make_position_embeddings(config, 10)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        for val in mgr.get_records()[0].per_head_delta:
            assert 0.0 <= val <= 2.0, f"Cosine distance out of range: {val}"

    def test_non_interference(self) -> None:
        """InstrumentManager must not change module output."""
        attn, config = _make_small_attn()
        torch.manual_seed(42)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)

        with torch.no_grad():
            out_clean, _ = attn(hidden_states, pos_emb, None)

        mgr = InstrumentManager()
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        with torch.no_grad():
            out_instrumented, _ = attn(hidden_states, pos_emb, None)

        torch.testing.assert_close(out_instrumented, out_clean, rtol=0, atol=0)

    def test_step_callback_compatible_with_logits_processor_list(self) -> None:
        """make_step_callback returns a LogitsProcessorList-compatible object."""
        from transformers import LogitsProcessorList

        mgr = InstrumentManager()
        callback = mgr.make_step_callback()

        processor_list = LogitsProcessorList([callback])
        assert len(processor_list) == 1
