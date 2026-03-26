# navi-SAD (Spectral Attention Divergence)

A dynamical systems probe for LLM inference. Runs softmax and linear attention in parallel on the same post-RoPE Q/K/V tensors, measures per-head cosine divergence, and reconstructs the model's internal attractor via delay-coordinate embedding (permutation entropy).

Confabulation detection remains one needle in the token-stack we are looking for, but it is no longer the only one. The instrument is hypothesis-agnostic -- it reconstructs dynamical structure. What you ask about that structure is a separate question.

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Method

SAD captures post-RoPE Q/K/V tensors from inside the model's native attention forward, then recomputes both softmax and linear attention in fp32. The cosine distance between per-head outputs produces a scalar trajectory over generation steps -- one time series per (layer, head) pair.

**Theoretical framing (Takens' embedding):** Each per-head SAD trajectory is treated as a delay-coordinate embedding of the model's internal dynamical state, following Takens' embedding theorem. We are not measuring a signal -- we are reconstructing an attractor. Permutation entropy is not a generic complexity heuristic here; it is load-bearing. Bandt-Pompe ordinal patterns are designed for exactly this: characterizing the ordinal structure of delay-coordinate reconstructions. When the attractor collapses (stereotyped dynamics, low PE), the model's internal state has lost the complex structure that characterizes one inference regime. When the attractor is rich (high PE, diverse ordinal patterns), the dynamics retain a different kind of structure. The per-head SAD trajectory is the observable; the attractor reconstruction is the instrument; PE on that reconstruction is the measurement.

**What the instrument can see:** Any regime that leaves a signature in per-head attention dynamics -- confidence, uncertainty, confabulation, creativity, rote retrieval, domain shifts, phase transitions during long-form generation. The first application is confabulation detection (TruthfulQA, correct vs incorrect attractor geometry), but the instrument does not hard-code that question.

**Scope limitation:** SAD is currently measured under cache-off conditions (`use_cache=False`), which forces full-prefix recomputation at each generation step. Generalization to cache-on (production) inference is unverified and remains a scope limitation.

## Research Grounding

The SAD instrument is theoretically motivated and adjacent-literature-grounded, but the attractor-reconstruction claim is not yet directly validated by repository evidence. Gates 0-2 validate the **instrument** (non-interference, parity, stability). Gate 3 tests whether the **reconstructed attractor carries information** -- starting with confabulation-relevant regimes but applicable to any labeled binary partition. The 40-sample pilot (completed) showed that grand-mean SAD does not separate correct from incorrect generations, but per-(layer, head) attractor structure and temporal dynamics remain open candidates.

**Theoretical basis -- softmax/linear capacity gap:**
Han et al. (2024, arXiv:2412.06590) prove that softmax attention is injective (different queries produce different distributions) while linear attention is not (distinct queries can collapse to identical outputs). This capacity gap is the structural basis for using softmax-linear divergence as a diagnostic. SAD does not claim that divergence directly measures truth -- it measures how much the model relies on its full nonlinear attention capacity versus operating in a regime where the weaker linear mechanism suffices.

**Adjacent empirical motifs:**
- D2HScore (Ding et al., 2025): low dispersion and drift in internal representations characterize hallucinated content.
- EigenTrack (arXiv:2509.15735, 2025): hallucinated sequences produce flatter, more dispersed attention spectra closer to the noise baseline.
- Neural Uncertainty Principle (arXiv:2603.19562, 2026): formalizes that weak prompt-gradient coupling indicates hallucination risk.
- Verbal uncertainty mismatch (EMNLP 2025, arXiv:2503.14477): the gap between high semantic uncertainty and low verbal uncertainty predicts hallucinations -- LLMs are overconfident when hallucinating.

**What is novel:** No published method runs two attention mechanisms in parallel on the same frozen weights as a dynamical systems probe. SAD combines known ingredients (linear attention, cosine divergence, delay-coordinate embedding via ordinal patterns) in a new configuration. The Takens framing -- treating per-head SAD as an attractor reconstruction rather than a scalar diagnostic -- is the theoretical contribution.

**What is not yet proven:** That the reconstructed attractors carry information about inference regimes rather than reflecting other sources of variation (prompt complexity, sequence length, topic domain). The 40-sample pilot showed the grand-mean signal washes out, but per-(layer, head) PE on first-differenced trajectories shows structural signal that survives across transforms. The open question is whether this structure corresponds to the computational-mechanical complexity of the inference problem -- specifically, whether per-head PE tracks the fractal dimension of the belief state attractor predicted by Shai et al. (NeurIPS 2024). Gate 3 tests this with synthetic HMM benchmarks where ground-truth fractal dimensions are known.

## Current State

- **Milestones A + B:** Complete. Core math, types, I/O, mock hooks, signal processing.
- **Milestone C:** Complete. Real instrumentation proven on Mistral-7B.
- **Milestone D (Gates 2-3):** Gate 2 passes. Gate 3 pilot complete (40 samples, 3-reviewer majority-vote labels). Key findings: grand-mean SAD does not separate groups; per-head PE on first-differenced trajectories shows structural signal (338/1024 heads with |d|>0.5 in 3+ mode/segment combos, 4.6:1 directional asymmetry). Full Gate 3 redesigned around synthetic HMM benchmarks with known fractal dimensions (see below).

249 tests (237 CPU + 12 GPU). CI enforces lint, format, typecheck, and test on every PR.

### Instrument Validation Summary

All validation performed on Mistral-7B-Instruct-v0.2 (fp16, eager attention, revision-pinned).

**Gate 0 -- Non-interference.** The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

**Gate 1 -- Parity.** Recomputed fp32 softmax attention, passed through the model's native o_proj, matches the native module output for the newest token. Calibrated across 2240 parity records (32 layers, short + medium sequences, 3 prompt shapes). Frozen thresholds: cosine similarity >= 0.999996 (worst observed: 0.99999869), relative L2 <= 0.00276 (worst observed: 0.00184). Pre-o_proj diagnostic confirms the error source is the expected fp32/fp16 precision asymmetry in the V matmul, not capture or projection bugs.

**Gate 2 -- Stability.** 50 consecutive generations with full instrumentation and JSONL serialization. Zero VRAM creep (0.0 MiB spread across 50 samples, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB). All 50 raw records round-trip through gzipped JSONL with intact provenance: schema fields, per-step/per-layer bijection, StepRecord type reconstruction, and all per-head cosine deltas finite and in [0, 2]. No graph retention, no memory leaks, no serialization drift.

## Scope (Phase 1)

- Model: Mistral-7B only (other families earn entry after Gate 1)
- Benchmark: Synthetic HMM family with known fractal dimensions (Gate 3). TruthfulQA as downstream application after instrument validation.
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
    pe_features.py    # SAD-specific PE: per-(layer,head) extraction, transforms, eligibility
    derivatives.py    # Finite differences on delta series
    aggregation.py    # Per-layer-per-head to per-token aggregation
  pilot/
    schema.py         # Typed write-side schema for pilot artifacts
    helpers.py        # Extraction, scoring, scalar computation, integrity validation
  io/
    writer.py         # Gzipped JSONL writer (raw records)
    reader.py         # Gzipped JSONL reader
    derived.py        # Derive analysis records from raw
tests/
  gates/              # GPU gate tests (@pytest.mark.gpu)
```

## Verification Gates

### Instrument validation (proven)

| Gate | What | Status |
|------|------|--------|
| 0 | Non-interference (identical tokens + logits with/without hooks) | **Passes** |
| 1 | Parity (recomputed fp32 softmax through o_proj matches native) | **Passes** |
| 2 | Memory stability (50 generations, no VRAM creep) | **Passes** |

### Hypothesis validation (in progress)

| Gate | What | Status |
|------|------|--------|
| 3 (pilot) | Per-head PE structure on TruthfulQA (40 samples) | **Complete** — structural signal found, grand-mean dead |
| 3 (full) | Rank correlation of per-head PE with known fractal dimension across synthetic HMM family | Planned |

See [ROADMAP.md](ROADMAP.md) for the full research plan.

## License

Apache-2.0. Copyright Project Navi LLC.
