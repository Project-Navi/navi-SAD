# navi-SAD (Spectral Attention Divergence)

Research harness for confabulation detection via dual-path attention comparison. Runs softmax and linear attention in parallel on the same post-RoPE Q/K/V tensors, measures per-head cosine divergence, and tracks temporal dynamics via ordinal patterns (permutation entropy).

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Method

SAD captures post-RoPE Q/K/V tensors from inside the model's native attention forward, then recomputes both softmax and linear attention in fp32. The cosine distance between per-head outputs is the core signal. Temporal dynamics are tracked via Bandt-Pompe ordinal patterns (permutation entropy) and finite differences on the per-token delta series.

Core hypothesis: when a model confabulates, the divergence between softmax and linear attention flatlines -- the model stops recruiting nonlinear attention capacity and coasts on smooth probabilistic flow.

## Current State

- **Milestones A + B:** Complete. Core math, types, I/O, mock hooks, signal processing.
- **Phases C1-C3:** Complete. MistralAdapter (Tier A forward-replacement), InstrumentManager, Gate 0 passes on Mistral-7B.
- **Phase C4:** In progress. Parity validation (Gate 1).

97 tests (91 CPU + 6 GPU). CI enforces lint, format, typecheck, and test on every PR.

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
    types.py          # StepRecord, RawSampleRecord, ModelFamilyConfig
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
| 0 | Non-interference (identical tokens + logits with/without hooks) | Passes |
| 1 | Parity (recomputed fp32 softmax through o_proj matches native) | In progress |
| 2 | Memory stability (50 generations, no VRAM creep) | Pending |
| 3 | Head sparsity (200 TruthfulQA samples, Cohen's d) | Pending |

## License

Apache-2.0. Copyright Project Navi LLC.
