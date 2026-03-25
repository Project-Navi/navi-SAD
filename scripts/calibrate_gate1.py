#!/usr/bin/env python3
"""One-off Gate 1 calibration. NOT part of CI.

Runs parity checks across layers and sequence lengths, reports
empirical tolerances. Review output manually, then freeze values
in tests/gates/test_gate1_parity.py.

Usage:
    uv run python scripts/calibrate_gate1.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.registry import get_family_config
from navi_sad.core.types import ParityConfig, ParityRecord


def _position_bucket(step_idx: int, max_new: int) -> str:
    """Classify a generation step into early/mid/late."""
    if step_idx <= 2:
        return "early"
    if step_idx >= max_new - 2:
        return "late"
    return "mid"


def main() -> None:
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    revision = "63a8b081895390a26e140280378bc85ec8bce07a"
    print(f"Loading {model_id} @ {revision[:12]}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    # Deterministic controls
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Prompts are hardcoded. They are not configurable parameters.
    prompts: dict[str, str] = {
        "short_plain": "The capital of France is",
        "short_structured": "Q: What is 2+2?\nA:",
        "medium_plain": (
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
        ),
    }
    max_new_tokens: dict[str, int] = {
        "short_plain": 20,
        "short_structured": 20,
        "medium_plain": 30,
    }

    # Verify tokenized lengths
    print("\nTokenized prompt lengths:")
    for label, prompt in prompts.items():
        tok_len = len(tokenizer.encode(prompt))
        print(f"  {label}: {tok_len} tokens")
        if "medium" in label:
            assert 128 <= tok_len <= 384, (
                f"{label} tokenized to {tok_len}, expected 128-384. Adjust prompt."
            )

    # Sliding window guard
    sliding_window = getattr(model.config, "sliding_window", None)
    if sliding_window is not None:
        for label, prompt in prompts.items():
            total_len = len(tokenizer.encode(prompt)) + max_new_tokens[label]
            assert total_len < sliding_window, (
                f"{label}: total sequence length {total_len} >= "
                f"sliding_window {sliding_window}. Reduce prompt or generation length."
            )
        print(f"  Sliding window: {sliding_window} (all sequences well under)")

    family = get_family_config(model.config)
    num_q = getattr(model.config, family.num_q_heads_attr)
    num_kv = getattr(model.config, family.num_kv_heads_attr)
    parity = ParityConfig(enabled=True, include_pre_oproj=True)

    all_records: list[ParityRecord] = []
    prompt_labels: list[str] = []
    step_max_new: list[int] = []

    for label, prompt in prompts.items():
        print(f"\n--- {label} ---")
        mgr = InstrumentManager(family, sink_exclude=0, parity=parity)

        for layer_idx in range(model.config.num_hidden_layers):
            attn_path = family.attn_module_path.format(layer_idx)
            attn_module = model.get_submodule(attn_path)
            mgr.install_layer(attn_module, layer_idx, num_q, num_kv)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_tok = max_new_tokens[label]
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=n_tok,
                min_new_tokens=n_tok,
                do_sample=False,
                use_cache=False,
                logits_processor=LogitsProcessorList([mgr.make_step_callback()]),
            )

        records = mgr.get_parity_records()
        all_records.extend(records)
        prompt_labels.extend([label] * len(records))
        step_max_new.extend([n_tok] * len(records))
        mgr.uninstall()

        cos_sims = [r.cosine_similarity for r in records]
        l2_errors = [r.relative_l2_error for r in records]
        max_abs = [r.max_absolute_error for r in records]

        print(f"  Records: {len(records)}")
        print(f"  Cosine sim:  min={min(cos_sims):.8f}  mean={sum(cos_sims) / len(cos_sims):.8f}")
        print(f"  Rel L2 err:  max={max(l2_errors):.8f}  mean={sum(l2_errors) / len(l2_errors):.8f}")
        print(f"  Max abs err: max={max(max_abs):.8f}  mean={sum(max_abs) / len(max_abs):.8f}")

        pre_cos = [r.pre_oproj_cosine for r in records if r.pre_oproj_cosine is not None]
        if pre_cos:
            print(f"  Pre-o_proj:  min={min(pre_cos):.8f}  mean={sum(pre_cos) / len(pre_cos):.8f}")

    # Aggregate
    print(f"\n=== AGGREGATE ({len(all_records)} records) ===")
    all_cos = [r.cosine_similarity for r in all_records]
    all_l2 = [r.relative_l2_error for r in all_records]
    all_abs = [r.max_absolute_error for r in all_records]

    print(f"  Cosine sim:  min={min(all_cos):.8f}")
    print(f"  Rel L2 err:  max={max(all_l2):.8f}")
    print(f"  Max abs err: max={max(all_abs):.8f}")

    # Worst cosine record
    worst_cos_idx = min(range(len(all_records)), key=lambda i: all_records[i].cosine_similarity)
    worst = all_records[worst_cos_idx]
    bucket = _position_bucket(worst.step_idx, step_max_new[worst_cos_idx])
    print(
        f"\n  Worst cosine: layer={worst.layer_idx} step={worst.step_idx} "
        f"prompt={prompt_labels[worst_cos_idx]} position={bucket} "
        f"cosine={worst.cosine_similarity:.8f}"
    )

    # Worst L2 record
    worst_l2_idx = max(range(len(all_records)), key=lambda i: all_records[i].relative_l2_error)
    worst_l2 = all_records[worst_l2_idx]
    bucket_l2 = _position_bucket(worst_l2.step_idx, step_max_new[worst_l2_idx])
    print(
        f"  Worst L2:     layer={worst_l2.layer_idx} step={worst_l2.step_idx} "
        f"prompt={prompt_labels[worst_l2_idx]} position={bucket_l2} "
        f"rel_l2={worst_l2.relative_l2_error:.8f}"
    )

    # Pre-o_proj diagnostic
    all_pre = [r.pre_oproj_cosine for r in all_records if r.pre_oproj_cosine is not None]
    if all_pre:
        print(f"\n  Pre-o_proj cosine: min={min(all_pre):.8f}")

    # Recommendation
    print("\n=== RECOMMENDATION ===")
    recommended_cosine = round(min(all_cos) - 0.0001, 4)
    recommended_l2 = round(max(all_l2) * 1.5, 6)
    print(f"  COSINE_MIN = {recommended_cosine}")
    print(f"  RELATIVE_L2_MAX = {recommended_l2}")
    print()
    print("  The script may recommend stricter thresholds than the provisional")
    print("  targets, but never weaker ones without an explicit adapter-fix review.")
    print()
    print("  Freeze these values in tests/gates/test_gate1_parity.py.")
    print("  NEVER relax them after observing benchmark results.")


if __name__ == "__main__":
    main()
