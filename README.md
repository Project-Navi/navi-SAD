# navi-SAD (Spectral Attention Divergence)

Research harness for confabulation detection via dual-path attention comparison. Runs softmax and linear attention in parallel on the same post-RoPE Q/K/V tensors, measures per-head cosine divergence, and tracks temporal dynamics via ordinal patterns (permutation entropy).

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Method

SAD captures post-RoPE Q/K/V tensors from inside the model's native attention forward, then recomputes both softmax and linear attention in fp32. The cosine distance between per-head outputs is the core signal. Temporal dynamics are tracked via Bandt-Pompe ordinal patterns (permutation entropy) and finite differences on the per-token delta series.

Core hypothesis: when a model confabulates, the divergence between softmax and linear attention flatlines -- the model stops recruiting nonlinear attention capacity and coasts on smooth probabilistic flow.

## Current State

- **Milestones A + B:** Complete. Core math, types, I/O, mock hooks, signal processing.
- **Milestone C:** Complete. Real instrumentation proven on Mistral-7B.
- **Milestone D (Gates 2-3):** Gate 2 passes. Gate 3 next (TruthfulQA head sparsity).

111 tests (99 CPU + 12 GPU). CI enforces lint, format, typecheck, and test on every PR.

### Instrument Validation Summary

All validation performed on Mistral-7B-Instruct-v0.2 (fp16, eager attention, revision-pinned).

**Gate 0 -- Non-interference.** The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

**Gate 1 -- Parity.** Recomputed fp32 softmax attention, passed through the model's native o_proj, matches the native module output for the newest token. Calibrated across 2240 parity records (32 layers, short + medium sequences, 3 prompt shapes). Frozen thresholds: cosine similarity >= 0.999996 (worst observed: 0.99999869), relative L2 <= 0.00276 (worst observed: 0.00184). Pre-o_proj diagnostic confirms the error source is the expected fp32/fp16 precision asymmetry in the V matmul, not capture or projection bugs.

**Gate 2 -- Stability.** 50 consecutive generations with full instrumentation and JSONL serialization. Zero VRAM creep (0.0 MiB spread across 50 samples, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB). All 50 raw records round-trip through gzipped JSONL with intact provenance: schema fields, per-step/per-layer bijection, StepRecord type reconstruction, and all per-head cosine deltas finite and in [0, 2]. No graph retention, no memory leaks, no serialization drift.

## Scope (Phase 1)

- Model: Mistral-7B only (other families earn entry after Gate 1)
- Benchmark: TruthfulQA generation (after Gate 3)
- No baselines until signal validated across architectures

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

For GPU gate tests, also install the eval dependencies:

```bash
uv sync --extra dev --extra eval
```

## Usage

```bash
# Lint + format + typecheck + CPU tests
make all

# GPU gate tests (requires CUDA + Mistral-7B weights)
make test-gpu

# Individual targets
make lint
make format
make typecheck
make test
```

## Project Structure

```
src/navi_sad/
  core/
    spectral.py       # Softmax + linear attention, GQA expansion, cosine distance
    hooks.py          # Mock hook manager (testing plumbing)
    adapter.py        # MistralAdapter (Tier A forward-replacement capture)
    instrument.py     # InstrumentManager (real model orchestration)
    registry.py       # Model family registry with adapter factory
    types.py          # StepRecord, RawSampleRecord, ModelFamilyConfig, ParityConfig, ParityRecord
  signal/
    ordinal.py        # Bandt-Pompe ordinal patterns, permutation entropy
    derivatives.py    # Finite differences on delta series
    aggregation.py    # Per-layer-per-head to per-token aggregation
  io/
    writer.py         # Gzipped JSONL writer (raw records)
    reader.py         # Gzipped JSONL reader
    derived.py        # Derive analysis records from raw
tests/
  gates/              # GPU gate tests (@pytest.mark.gpu)
```

## Verification Gates

| Gate | What | Status |
|------|------|--------|
| 0 | Non-interference (identical tokens + logits with/without hooks) | **Passes** |
| 1 | Parity (recomputed fp32 softmax through o_proj matches native) | **Passes** |
| 2 | Memory stability (50 generations, no VRAM creep) | **Passes** |
| 3 | Head sparsity (200 TruthfulQA samples, Cohen's d) | Pending |

## License

Apache-2.0. Copyright Project Navi LLC.
