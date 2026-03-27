# Gate 2 --- Stability

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

50 consecutive generations with full instrumentation and JSONL serialization. Zero VRAM creep (0.0 MiB spread, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB). Provenance round-trip validated.

See [Gate Discipline](gate-discipline.md) for the stability methodology and [I/O modules](../reference/module-reference.md) for the JSONL serialization pipeline.

---

## Protocol

2 warmup generations followed by 50 measured generations on a fixed long-form probe prompt, with full instrumentation and JSONL serialization active. Each generation: 100 tokens, greedy decode, cache off.

### Warmup discipline

The first 2 generations exercise the full pipeline but are excluded from gate metrics and JSONL output. This prevents CUDA lazy initialization and JIT compilation from inflating the baseline.

### Baseline and thresholds

VRAM baseline: `max(vram[:3])` --- the maximum post-cleanup allocation across the first 3 measured samples. Using the max rather than the mean guards against a low first-measurement biasing the baseline. If early samples are unstable, increase warmup --- never the threshold.

Gate 2 uses **fixed thresholds** (engineering limits, not calibrated from observed behavior):

| Metric | Threshold | Observed |
|--------|-----------|----------|
| VRAM spread (`max - min`) | <= 16 MiB | 0.0 MiB |
| CPU RSS growth (`rss[-1] - rss[0]`) | <= 128 MiB | 0.7 MiB |

## Memory measurement

**VRAM:** After each generation, explicit cleanup (`del output, records`, `mgr.reset()`, `gc.collect()`, `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`). Post-cleanup `torch.cuda.memory_allocated()` is the gate metric. `reset_peak_memory_stats()` called before each generation for per-sample peaks.

**CPU RSS:** `psutil.Process(os.getpid()).memory_info().rss` post-cleanup. The growth check catches monotonic Python-side leaks (accumulating objects, uncollected records, buffer growth) that would not appear in VRAM.

## Provenance checks

The second Gate 2 test reads back the written JSONL and validates every record:

- **Schema fields:** `schema_version == 1`, `record_type == "raw"`, correct model ID, `capture_tier == "A"`, `benchmark == "gate2_stability"`, `num_tokens_generated == 100`, `layers_hooked == [0..31]`
- **Record count:** Exactly 50 (warmup excluded)
- **Sample ID uniqueness:** No duplicates
- **Per-step/per-layer bijection:** Same structural invariant as [Gate 0](gate-0.md) --- each step has exactly one entry per layer, no duplicates, no gaps
- **StepRecord reconstruction:** First `per_step` entry passed to `StepRecord(**entry)` to verify schema compatibility between write and read paths
- **Delta range \( [0, 2] \):** Every `per_head_delta` value must be finite and within \( [0.0, 2.0] \). Cosine distance is bounded by this range. Values outside indicate NaN, Inf, or numerical corruption
