"""Gate 1: Family Parity.

Recomputed fp32 softmax through native o_proj must match native output.

Tolerances are FROZEN from calibration output (scripts/calibrate_gate1.py).
Never relax them. If a test fails, the adapter is broken -- fix the
adapter, not the threshold.

Calibration results (2240 records on Mistral-7B-Instruct-v0.2):
  Cosine sim:  min=0.99999869
  Rel L2 err:  max=0.00183902

Spec: SPEC.md sections 5.3, 9 (Gate 1).
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from transformers import LogitsProcessorList

from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.registry import get_family_config
from navi_sad.core.types import ParityConfig

# === FROZEN TOLERANCES ===
# Set from scripts/calibrate_gate1.py output on 2026-03-24.
# NEVER relax after observing benchmark results.
#
# Cosine frozen in distance-from-1 space:
#   worst observed delta = 1.31e-6, 3x headroom -> 3.93e-6
#   COSINE_MIN = 1 - 3.93e-6 = 0.999996
COSINE_MIN: float = 0.999996
RELATIVE_L2_MAX: float = 0.002759
# =========================


@pytest.mark.gpu
class TestGate1Parity:
    SHORT_PROMPTS: ClassVar[list[tuple[str, str]]] = [
        ("short_plain", "The capital of France is"),
        ("short_structured", "Q: What is 2+2?\nA:"),
    ]

    MEDIUM_PROMPT: ClassVar[str] = (
        "Explain in thorough detail how the process of photosynthesis "
        "works in plants. Cover the light-dependent reactions that "
        "occur in the thylakoid membranes, including the role of "
        "photosystems I and II, the electron transport chain, and "
        "chemiosmosis. Then explain the Calvin cycle that occurs in "
        "the stroma, including carbon fixation by RuBisCO, the "
        "reduction phase, and regeneration of RuBP. Discuss the role "
        "of chlorophyll a, chlorophyll b, and accessory pigments like "
        "carotenoids in absorbing different wavelengths of light. "
        "Finally, explain how environmental factors such as light "
        "intensity, CO2 concentration, and temperature affect the "
        "overall rate of photosynthesis in C3 and C4 plants."
    )

    def _install_parity(self, model) -> InstrumentManager:  # type: ignore[no-untyped-def]
        family = get_family_config(model.config)
        parity = ParityConfig(enabled=True, include_pre_oproj=True)
        mgr = InstrumentManager(family, sink_exclude=0, parity=parity)
        num_q = getattr(model.config, family.num_q_heads_attr)
        num_kv = getattr(model.config, family.num_kv_heads_attr)

        for layer_idx in range(model.config.num_hidden_layers):
            attn_path = family.attn_module_path.format(layer_idx)
            attn_module = model.get_submodule(attn_path)
            mgr.install_layer(attn_module, layer_idx, num_q, num_kv)

        return mgr

    def _run_parity(self, model, tokenizer, prompt, max_new):  # type: ignore[no-untyped-def]
        mgr = self._install_parity(model)
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Enforce unpadded input -- parity closure does not use attention mask
            assert "attention_mask" in inputs
            assert inputs["attention_mask"].all().item(), "Padded input detected"
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    min_new_tokens=max_new,
                    do_sample=False,
                    use_cache=False,
                    logits_processor=LogitsProcessorList([mgr.make_step_callback()]),
                )
        finally:
            records = mgr.get_parity_records()
            mgr.uninstall()
        return records

    @pytest.mark.parametrize(
        "label,prompt",
        SHORT_PROMPTS,
        ids=[p[0] for p in SHORT_PROMPTS],
    )
    def test_parity_short_sequences(
        self,
        mistral_model_and_tokenizer,  # type: ignore[no-untyped-def]
        label: str,
        prompt: str,
    ) -> None:
        model, tokenizer = mistral_model_and_tokenizer
        records = self._run_parity(model, tokenizer, prompt, max_new=20)

        assert len(records) > 0, f"No parity records for {label}"
        for r in records:
            assert r.cosine_similarity >= COSINE_MIN, (
                f"Gate 1 FAILED ({label}): layer={r.layer_idx} step={r.step_idx} "
                f"cosine={r.cosine_similarity:.8f} < {COSINE_MIN} "
                f"(max_abs={r.max_absolute_error:.8f})"
            )
            assert r.relative_l2_error <= RELATIVE_L2_MAX, (
                f"Gate 1 FAILED ({label}): layer={r.layer_idx} step={r.step_idx} "
                f"rel_l2={r.relative_l2_error:.8f} > {RELATIVE_L2_MAX} "
                f"(max_abs={r.max_absolute_error:.8f})"
            )

    def test_parity_medium_sequence(self, mistral_model_and_tokenizer) -> None:  # type: ignore[no-untyped-def]
        """Spec requires medium (~256 token) sequences.

        Asserts tokenized prompt length is in the medium range and
        total sequence stays below sliding window.
        """
        model, tokenizer = mistral_model_and_tokenizer
        max_new = 30

        inputs = tokenizer(self.MEDIUM_PROMPT, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        assert 128 <= prompt_len <= 384, (
            f"Medium prompt tokenized to {prompt_len} tokens, expected 128-384."
        )

        # Sliding window guard
        sliding_window = getattr(model.config, "sliding_window", None)
        if sliding_window is not None:
            assert prompt_len + max_new < sliding_window, (
                f"Total length {prompt_len + max_new} >= sliding_window {sliding_window}"
            )

        records = self._run_parity(model, tokenizer, self.MEDIUM_PROMPT, max_new=max_new)

        assert len(records) > 0
        for r in records:
            assert r.cosine_similarity >= COSINE_MIN, (
                f"Gate 1 FAILED (medium): layer={r.layer_idx} step={r.step_idx} "
                f"cosine={r.cosine_similarity:.8f}"
            )
            assert r.relative_l2_error <= RELATIVE_L2_MAX, (
                f"Gate 1 FAILED (medium): layer={r.layer_idx} step={r.step_idx} "
                f"rel_l2={r.relative_l2_error:.8f}"
            )

    def test_no_systematic_layer_drift(self, mistral_model_and_tokenizer) -> None:  # type: ignore[no-untyped-def]
        """No early/mid/late layer's mean cosine should fall below the frozen global minimum."""
        model, tokenizer = mistral_model_and_tokenizer
        num_layers = model.config.num_hidden_layers

        records = self._run_parity(
            model,
            tokenizer,
            "The capital of France is",
            max_new=10,
        )

        by_layer: dict[int, list[float]] = {}
        for r in records:
            by_layer.setdefault(r.layer_idx, []).append(r.cosine_similarity)

        for layer_idx in [0, num_layers // 2, num_layers - 1]:
            sims = by_layer.get(layer_idx, [])
            if sims:
                mean_sim = sum(sims) / len(sims)
                assert mean_sim >= COSINE_MIN, (
                    f"Layer drift: layer {layer_idx} mean cosine={mean_sim:.8f} < {COSINE_MIN}"
                )
