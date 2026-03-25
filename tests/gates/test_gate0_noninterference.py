"""Gate 0: Non-Interference.

Identical output tokens AND allclose logits under deterministic decoding
with and without instrumentation. If this fails, the observer is
perturbing the system. Do not proceed.

Spec: SPEC.md section 9, Gate 0.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch
from transformers import LogitsProcessorList

from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.registry import get_family_config


@pytest.mark.gpu
class TestGate0NonInterference:
    PROMPTS: ClassVar[list[str]] = [
        "The capital of France is",
        "def fibonacci(n):\n    '''Return the nth Fibonacci number.'''\n",
        "Explain quantum entanglement in simple terms:",
    ]
    MAX_NEW_TOKENS = 30

    def _install_instrument(self, model) -> InstrumentManager:  # type: ignore[no-untyped-def]
        family = get_family_config(model.config)
        mgr = InstrumentManager(sink_exclude=1)
        num_q = getattr(model.config, family.num_q_heads_attr)
        num_kv = getattr(model.config, family.num_kv_heads_attr)

        for layer_idx in range(model.config.num_hidden_layers):
            attn_path = family.attn_module_path.format(layer_idx)
            attn_module = model.get_submodule(attn_path)
            mgr.install_layer(attn_module, layer_idx, num_q, num_kv)

        return mgr

    def _generate(self, model, tokenizer, prompt, max_new, mgr=None):  # type: ignore[no-untyped-def]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs: dict = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "use_cache": False,
        }
        if mgr is not None:
            gen_kwargs["logits_processor"] = LogitsProcessorList([mgr.make_step_callback()])
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        return output[0].cpu()

    def test_baseline_determinism(self, mistral_model_and_tokenizer) -> None:  # type: ignore[no-untyped-def]
        """Model must produce identical tokens across two greedy runs."""
        model, tokenizer = mistral_model_and_tokenizer
        tokens_a = self._generate(model, tokenizer, self.PROMPTS[0], self.MAX_NEW_TOKENS)
        tokens_b = self._generate(model, tokenizer, self.PROMPTS[0], self.MAX_NEW_TOKENS)
        assert torch.equal(tokens_a, tokens_b), (
            "Model is non-deterministic without hooks. Cannot proceed."
        )

    @pytest.mark.parametrize("prompt", PROMPTS, ids=["factual", "code", "explain"])
    def test_identical_tokens(self, mistral_model_and_tokenizer, prompt) -> None:  # type: ignore[no-untyped-def]
        """Output tokens must be identical with and without hooks."""
        model, tokenizer = mistral_model_and_tokenizer
        tokens_clean = self._generate(model, tokenizer, prompt, self.MAX_NEW_TOKENS)

        mgr = self._install_instrument(model)
        try:
            tokens_hooked = self._generate(model, tokenizer, prompt, self.MAX_NEW_TOKENS, mgr=mgr)
        finally:
            mgr.uninstall()

        assert torch.equal(tokens_clean, tokens_hooked), (
            f"Gate 0 FAILED: hooks changed output tokens.\n"
            f"Clean:  {tokenizer.decode(tokens_clean)}\n"
            f"Hooked: {tokenizer.decode(tokens_hooked)}"
        )

    def test_logits_exact_match(self, mistral_model_and_tokenizer) -> None:  # type: ignore[no-untyped-def]
        """Single-forward logit comparison. Token identity alone is insufficient."""
        model, tokenizer = mistral_model_and_tokenizer
        inputs = tokenizer(self.PROMPTS[0], return_tensors="pt").to(model.device)

        with torch.no_grad():
            logits_clean = model(**inputs).logits.cpu()

        mgr = self._install_instrument(model)
        try:
            with torch.no_grad():
                logits_hooked = model(**inputs).logits.cpu()
        finally:
            mgr.uninstall()

        torch.testing.assert_close(
            logits_hooked,
            logits_clean,
            rtol=0,
            atol=0,
            msg="Gate 0 FAILED: hooks perturbed logits",
        )

    def test_record_count_matches_contract(self, mistral_model_and_tokenizer) -> None:  # type: ignore[no-untyped-def]
        """Verify step accounting: exactly num_layers * max_new_tokens records.

        Uses min_new_tokens = max_new_tokens to force exactly N forwards.
        Without this, early EOS would reduce the count and look like
        broken step accounting.
        """
        model, tokenizer = mistral_model_and_tokenizer
        max_new = 5

        mgr = self._install_instrument(model)
        try:
            inputs = tokenizer(self.PROMPTS[0], return_tensors="pt").to(model.device)
            gen_kwargs: dict = {
                "max_new_tokens": max_new,
                "min_new_tokens": max_new,
                "do_sample": False,
                "use_cache": False,
                "logits_processor": LogitsProcessorList([mgr.make_step_callback()]),
            }
            with torch.no_grad():
                model.generate(**inputs, **gen_kwargs)
        finally:
            records = mgr.get_records()
            mgr.uninstall()

        num_layers = model.config.num_hidden_layers
        expected = num_layers * max_new
        assert len(records) == expected, (
            f"Step accounting broken: expected {expected} records "
            f"({num_layers} layers x {max_new} tokens), got {len(records)}"
        )

        num_q = model.config.num_attention_heads
        for r in records:
            assert len(r.per_head_delta) == num_q
