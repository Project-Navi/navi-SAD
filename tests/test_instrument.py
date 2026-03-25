"""Tests for InstrumentManager -- real model orchestration."""

from __future__ import annotations

import torch
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
)

from navi_sad.core.adapter import MistralAdapter
from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.types import ModelFamilyConfig, ParityConfig

_CPU = torch.device("cpu")

# Minimal family config for CPU tests — only adapter_factory matters here.
_TEST_FAMILY = ModelFamilyConfig(
    architecture="MistralForCausalLM",
    attn_module_path="model.layers.{}.self_attn",
    capture_tier="A",
    num_kv_heads_attr="num_key_value_heads",
    num_q_heads_attr="num_attention_heads",
    head_dim_attr="head_dim",
    gqa_expansion=True,
    adapter_factory=MistralAdapter,
)


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


class TestInstrumentManagerInit:
    def test_adapter_factory_none_raises(self) -> None:
        """InstrumentManager rejects family config without adapter_factory."""
        import pytest

        bad_config = ModelFamilyConfig(
            architecture="TestArch",
            attn_module_path="model.layers.{}.self_attn",
            capture_tier="A",
            num_kv_heads_attr="num_key_value_heads",
            num_q_heads_attr="num_attention_heads",
            head_dim_attr="head_dim",
            gqa_expansion=True,
            adapter_factory=None,
        )
        with pytest.raises(ValueError, match="no adapter_factory"):
            InstrumentManager(bad_config)


class TestInstrumentManager:
    def test_single_layer_produces_records(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=1)
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
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0)
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
        mgr = InstrumentManager(_TEST_FAMILY)
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
        mgr = InstrumentManager(_TEST_FAMILY)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)
        mgr.uninstall()

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)

        assert len(mgr.get_records()) == 0

    def test_per_head_delta_in_valid_range(self) -> None:
        attn, config = _make_small_attn()
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0)
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

        mgr = InstrumentManager(_TEST_FAMILY)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        with torch.no_grad():
            out_instrumented, _ = attn(hidden_states, pos_emb, None)

        torch.testing.assert_close(out_instrumented, out_clean, rtol=0, atol=0)

    def test_step_callback_compatible_with_logits_processor_list(self) -> None:
        """make_step_callback returns a LogitsProcessorList-compatible object."""
        from transformers import LogitsProcessorList

        mgr = InstrumentManager(_TEST_FAMILY)
        callback = mgr.make_step_callback()

        processor_list = LogitsProcessorList([callback])
        assert len(processor_list) == 1

    def test_step_callback_increments_step_idx(self) -> None:
        """Calling the step callback actually increments the step counter."""
        mgr = InstrumentManager(_TEST_FAMILY)
        callback = mgr.make_step_callback()

        assert mgr._step_idx == 0
        dummy_ids = torch.zeros(1, 5, dtype=torch.long)
        dummy_scores = torch.zeros(1, 100)
        callback(dummy_ids, dummy_scores)
        assert mgr._step_idx == 1
        callback(dummy_ids, dummy_scores)
        assert mgr._step_idx == 2


class TestInstrumentManagerParity:
    def test_parity_mode_produces_records(self) -> None:
        attn, config = _make_small_attn()
        parity = ParityConfig(enabled=True, include_pre_oproj=True)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        parity_records = mgr.get_parity_records()
        assert len(parity_records) == 1
        assert parity_records[0].layer_idx == 0
        assert parity_records[0].step_idx == 0
        assert 0.0 <= parity_records[0].cosine_similarity <= 1.0
        assert parity_records[0].relative_l2_error >= 0.0
        assert parity_records[0].max_absolute_error >= 0.0
        assert parity_records[0].pre_oproj_cosine is not None

    def test_parity_mha_no_expansion(self) -> None:
        """Parity works when num_q_heads == num_kv_heads (MHA, no GQA expansion)."""
        attn, config = _make_small_attn(hidden_size=64, num_heads=4, num_kv_heads=4)
        parity = ParityConfig(enabled=True, include_pre_oproj=True)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=4)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        records = mgr.get_parity_records()
        assert len(records) == 1
        assert 0.0 <= records[0].cosine_similarity <= 1.0

    def test_parity_none_produces_no_records(self) -> None:
        """parity=None means no parity at all."""
        attn, config = _make_small_attn()
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        assert len(mgr.get_parity_records()) == 0

    def test_parity_enabled_false_produces_no_records(self) -> None:
        """ParityConfig(enabled=False) also means no parity."""
        attn, config = _make_small_attn()
        parity = ParityConfig(enabled=False)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        assert len(mgr.get_parity_records()) == 0

    def test_parity_without_pre_oproj(self) -> None:
        attn, config = _make_small_attn()
        parity = ParityConfig(enabled=True, include_pre_oproj=False)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        records = mgr.get_parity_records()
        assert len(records) == 1
        assert records[0].pre_oproj_cosine is None

    def test_reset_clears_parity_records(self) -> None:
        attn, config = _make_small_attn()
        parity = ParityConfig(enabled=True)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)
        with torch.no_grad():
            attn(hidden_states, pos_emb, None)
        mgr.step()

        assert len(mgr.get_parity_records()) == 1
        mgr.reset()
        assert len(mgr.get_parity_records()) == 0

    def test_parity_non_interference(self) -> None:
        """Parity mode must not change module output."""
        attn, config = _make_small_attn()
        torch.manual_seed(42)

        hidden_states = torch.randn(1, 8, 64)
        pos_emb = _make_position_embeddings(config, 8)

        with torch.no_grad():
            out_clean, _ = attn(hidden_states, pos_emb, None)

        parity = ParityConfig(enabled=True)
        mgr = InstrumentManager(_TEST_FAMILY, sink_exclude=0, parity=parity)
        mgr.install_layer(attn, layer_idx=0, num_q_heads=4, num_kv_heads=2)

        with torch.no_grad():
            out_parity, _ = attn(hidden_states, pos_emb, None)

        torch.testing.assert_close(out_parity, out_clean, rtol=0, atol=0)
