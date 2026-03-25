"""Gate 2: Serialization / Stability.

Proves the instrument does not leak memory or lose data across 50
consecutive generations on Mistral-7B. This is the last mechanical
validation before benchmark runs (Gate 3).

Fixed thresholds (not calibrated from observed behavior):
  VRAM post-cleanup drift: 16 MiB
  CPU RSS growth: 128 MiB

Spec: SPEC.md section 9 (Gate 2), docs/plans/GATE2_SPEC.md.
"""

from __future__ import annotations

import gc
import math
import os
from uuid import uuid4

import psutil
import pytest
import torch
from transformers import LogitsProcessorList

from navi_sad.core.instrument import InstrumentManager
from navi_sad.core.registry import get_family_config
from navi_sad.core.types import RawSampleRecord, StepRecord
from navi_sad.io.reader import RawRecordReader
from navi_sad.io.writer import RawRecordWriter

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
MAX_NEW_TOKENS = 100
NUM_WARMUP = 2
NUM_MEASURED = 50

# Fixed thresholds -- not calibrated from observed behavior
VRAM_TOLERANCE_BYTES = 16 * 1024 * 1024  # 16 MiB
CPU_RSS_TOLERANCE_BYTES = 128 * 1024 * 1024  # 128 MiB

PROBE_PROMPT = (
    "Describe the complete process of how a bill becomes a law in the "
    "United States, starting from the initial drafting stage through "
    "committee review, floor debate, reconciliation between chambers, "
    "and presidential action. Include the roles of the Speaker of the "
    "House, the Senate Majority Leader, and the various standing "
    "committees in shaping legislation."
)


@pytest.fixture(scope="module")
def gate2_results(mistral_model_and_tokenizer, tmp_path_factory):  # type: ignore[no-untyped-def]
    """Run 2 warmup + 50 measured generations, yield results for both tests."""
    model, tokenizer = mistral_model_and_tokenizer

    # Tokenize and validate prompt
    inputs = tokenizer(PROBE_PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    assert 50 <= prompt_len <= 100, f"Prompt tokenized to {prompt_len}, expected 50-100"
    assert "attention_mask" in inputs
    assert inputs["attention_mask"].all().item(), "Padded input detected"
    sliding_window = getattr(model.config, "sliding_window", None)
    if sliding_window is not None:
        assert prompt_len + MAX_NEW_TOKENS < sliding_window

    # Build InstrumentManager from registry
    family = get_family_config(model.config)
    mgr = InstrumentManager(family, sink_exclude=1)
    num_q = getattr(model.config, family.num_q_heads_attr)
    num_kv = getattr(model.config, family.num_kv_heads_attr)
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        attn_path = family.attn_module_path.format(layer_idx)
        attn_module = model.get_submodule(attn_path)
        mgr.install_layer(attn_module, layer_idx, num_q, num_kv)

    # Step callback created once, reused across all samples
    step_callback = mgr.make_step_callback()

    # Writer + run ID
    jsonl_path = tmp_path_factory.mktemp("gate2") / "raw.jsonl.gz"
    writer = RawRecordWriter(jsonl_path)
    run_id = f"gate2_{uuid4().hex[:8]}"

    process = psutil.Process(os.getpid())
    vram_post_cleanup: list[int] = []
    rss_post_cleanup: list[int] = []
    diag_pre_cleanup: list[dict] = []
    written_sample_ids: list[str] = []

    try:
        for i in range(NUM_WARMUP + NUM_MEASURED):
            is_warmup = i < NUM_WARMUP
            output = None
            records = None
            raw_record = None

            try:
                torch.cuda.reset_peak_memory_stats()

                # Generate
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_cache=False,
                        logits_processor=LogitsProcessorList([step_callback]),
                    )

                # Assert actual generated token count
                generated_len = output.shape[1] - inputs["input_ids"].shape[1]
                assert generated_len == MAX_NEW_TOKENS, (
                    f"Expected {MAX_NEW_TOKENS} generated tokens, got {generated_len}"
                )

                # Collect records
                records = mgr.get_records()
                sample_id = f"gate2_{i:03d}"

                # Only write measured samples -- spec says exactly 50 records
                if not is_warmup:
                    raw_record = RawSampleRecord(
                        sample_id=sample_id,
                        model=MODEL_ID,
                        benchmark="gate2_stability",
                        prompt=PROBE_PROMPT,
                        generation=tokenizer.decode(
                            output[0][inputs["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        ),
                        num_tokens_generated=MAX_NEW_TOKENS,
                        layers_hooked=list(range(num_layers)),
                        capture_tier="A",
                        per_step=records,
                        metadata={
                            "run_id": run_id,
                            "revision": REVISION,
                        },
                    )
                    writer.write(raw_record)
                    written_sample_ids.append(sample_id)

                # Diagnostics (post-generation, pre-cleanup)
                torch.cuda.synchronize()
                diag_pre_cleanup.append(
                    {
                        "allocated": torch.cuda.memory_allocated(),
                        "peak": torch.cuda.max_memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                    }
                )

            finally:
                # Cleanup ALWAYS runs, even if generation/assertion fails
                del output, records, raw_record
                mgr.reset()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Gate metrics (post-cleanup)
            if not is_warmup:
                vram_post_cleanup.append(torch.cuda.memory_allocated())
                rss_post_cleanup.append(process.memory_info().rss)

        # Close writer BEFORE yielding -- provenance test reads the file
        writer.close()

        yield {
            "vram_post_cleanup": vram_post_cleanup,
            "rss_post_cleanup": rss_post_cleanup,
            "diag_pre_cleanup": diag_pre_cleanup,
            "written_sample_ids": written_sample_ids,
            "jsonl_path": jsonl_path,
            "run_id": run_id,
            "num_layers": num_layers,
            "num_q_heads": num_q,
        }

    finally:
        mgr.uninstall()


@pytest.mark.gpu
def test_no_vram_creep(gate2_results) -> None:  # type: ignore[no-untyped-def]
    """VRAM must not creep across 50 generations."""
    vram = gate2_results["vram_post_cleanup"]
    rss = gate2_results["rss_post_cleanup"]

    # Baseline = max of first 3 measured samples
    baseline = max(vram[:3])

    # Every sample within 16 MiB of baseline
    for i, v in enumerate(vram):
        assert abs(v - baseline) <= VRAM_TOLERANCE_BYTES, (
            f"VRAM drift at sample {i}: {v} vs baseline {baseline} "
            f"(delta={v - baseline} bytes, limit={VRAM_TOLERANCE_BYTES})"
        )

    # Spread within 16 MiB
    spread = max(vram) - min(vram)
    assert spread <= VRAM_TOLERANCE_BYTES, f"VRAM spread {spread} bytes > {VRAM_TOLERANCE_BYTES}"

    # CPU RSS growth within 128 MiB
    rss_growth = rss[-1] - rss[0]
    assert rss_growth <= CPU_RSS_TOLERANCE_BYTES, (
        f"CPU RSS grew {rss_growth} bytes > {CPU_RSS_TOLERANCE_BYTES}"
    )

    # Diagnostic summary
    print("\n=== Gate 2 Memory Summary ===")
    print(f"  VRAM baseline: {baseline / 1024**2:.1f} MiB")
    print(
        f"  VRAM spread:   {spread / 1024**2:.1f} MiB (limit: {VRAM_TOLERANCE_BYTES / 1024**2:.0f} MiB)"
    )
    print(
        f"  CPU RSS growth: {rss_growth / 1024**2:.1f} MiB (limit: {CPU_RSS_TOLERANCE_BYTES / 1024**2:.0f} MiB)"
    )


@pytest.mark.gpu
def test_provenance_round_trip(gate2_results) -> None:  # type: ignore[no-untyped-def]
    """All 50 records survive JSONL round-trip with intact provenance."""
    jsonl_path = gate2_results["jsonl_path"]
    run_id = gate2_results["run_id"]
    expected_ids = gate2_results["written_sample_ids"]
    num_layers = gate2_results["num_layers"]

    measured_records = list(RawRecordReader(jsonl_path))

    # Exactly 50 records (warmup samples are not written)
    assert len(measured_records) == NUM_MEASURED, (
        f"Expected {NUM_MEASURED} records, got {len(measured_records)}"
    )

    seen_ids: set[str] = set()

    for r in measured_records:
        # Uniqueness
        assert r["sample_id"] not in seen_ids, f"Duplicate sample_id: {r['sample_id']}"
        seen_ids.add(r["sample_id"])

        # Schema fields
        assert r["schema_version"] == 1
        assert r["record_type"] == "raw"
        assert r["model"] == MODEL_ID
        assert r["capture_tier"] == "A"
        assert r["benchmark"] == "gate2_stability"
        assert r["num_tokens_generated"] == MAX_NEW_TOKENS
        assert len(r["layers_hooked"]) == num_layers
        assert r["metadata"]["run_id"] == run_id
        assert r["metadata"]["revision"] == REVISION

        # Per-step bijection (reuses Gate 0 contract)
        per_step = r["per_step"]
        assert len(per_step) == num_layers * MAX_NEW_TOKENS, (
            f"Expected {num_layers * MAX_NEW_TOKENS} per_step entries, got {len(per_step)}"
        )

        by_step: dict[int, list[int]] = {}
        for s in per_step:
            by_step.setdefault(s["step_idx"], []).append(s["layer_idx"])

        assert set(by_step.keys()) == set(range(MAX_NEW_TOKENS)), (
            f"Expected step_idx 0..{MAX_NEW_TOKENS - 1}, got {sorted(by_step.keys())}"
        )
        expected_layers = set(range(num_layers))
        for step_idx, layer_indices in by_step.items():
            assert set(layer_indices) == expected_layers, (
                f"Step {step_idx}: expected layers {expected_layers}, got {set(layer_indices)}"
            )
            assert len(layer_indices) == num_layers, (
                f"Step {step_idx}: duplicate layers: {layer_indices}"
            )

        # StepRecord reconstruction
        first = per_step[0]
        reconstructed = StepRecord(**first)
        assert reconstructed.step_idx == first["step_idx"]

        # Delta value range -- catches NaN/Inf/corruption
        for s in per_step:
            for val in s["per_head_delta"]:
                assert math.isfinite(val), f"Non-finite delta: {val}"
                assert 0.0 <= val <= 2.0, f"Delta out of range: {val}"

    assert seen_ids == set(expected_ids), "sample_id mismatch between written and read"
