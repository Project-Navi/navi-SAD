# navi-SAD (Spectral Attention Divergence)

A dynamical systems probe for LLM inference. Runs softmax and linear attention in parallel on the same post-RoPE Q/K/V tensors, measures per-head cosine divergence, and reconstructs the model's internal attractor via delay-coordinate embedding (permutation entropy).

navi-SAD is a runtime measurement instrument for transformer inference, not an application. It captures per-head softmax-vs-linear attention divergence as a delay-coordinate embedding of the model's residual-stream dynamics. Whether that embedding carries information about inference regime is an open empirical question. Validation is in progress on synthetic HMM benchmarks (Gate 3); the repository makes no application claims today.

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Method

SAD captures post-RoPE Q/K/V tensors from inside the model's native attention forward, then recomputes both softmax and linear attention in fp32. The cosine distance between per-head outputs produces a scalar trajectory over generation steps -- one time series per (layer, head) pair.

**Theoretical framing (Takens' embedding):** Each per-head SAD trajectory is treated as a delay-coordinate embedding of the model's internal dynamical state, following Takens' embedding theorem. We are not measuring a signal -- we are reconstructing an attractor. Permutation entropy is not a generic complexity heuristic here; it is load-bearing. Bandt-Pompe ordinal patterns are designed for exactly this: characterizing the ordinal structure of delay-coordinate reconstructions. When the attractor collapses (stereotyped dynamics, low PE), the model's internal state has lost the complex structure that characterizes one inference regime. When the attractor is rich (high PE, diverse ordinal patterns), the dynamics retain a different kind of structure. The per-head SAD trajectory is the observable; the attractor reconstruction is the instrument; PE on that reconstruction is the measurement.

**What the instrument outputs:** A per-(layer, head) scalar trajectory -- the cosine divergence between softmax and linear attention on the same post-RoPE Q/K/V tensors -- over generation steps. We treat that trajectory as a delay-coordinate observable of the residual-stream state. What can be read off it is an open empirical question.

**Scope limitation:** SAD is currently measured under cache-off conditions (`use_cache=False`), which forces full-prefix recomputation at each generation step. Generalization to cache-on (production) inference is unverified and remains a scope limitation.

## Research Grounding

Gates 0-2 validate the **instrument** (non-interference, parity, stability). The 40-sample TruthfulQA pilot and the 400-sample replication are now closed methodological case studies; the dense-d direction observed in the pilot did not survive a length-matched permutation null (p=0.96). Gate 3 has been redesigned around synthetic HMM benchmarks with known fractal dimensions.

**Theoretical basis -- softmax/linear capacity gap:**
Han et al. (2024, arXiv:2412.06590) prove that softmax attention is injective (different queries produce different distributions) while linear attention is not (distinct queries can collapse to identical outputs). This capacity gap is the structural basis for using softmax-linear divergence as a diagnostic. SAD does not claim that divergence directly measures truth -- it measures how much the model relies on its full nonlinear attention capacity versus operating in a regime where the weaker linear mechanism suffices.

**Related representation-dynamics work (cited for context, not as application claims):**
- D2HScore (Ding et al., 2025): low dispersion and drift in internal representations characterize hallucinated content.
- EigenTrack (arXiv:2509.15735, 2025): hallucinated sequences produce flatter, more dispersed attention spectra closer to the noise baseline.
- Neural Uncertainty Principle (arXiv:2603.19562, 2026): formalizes that weak prompt-gradient coupling indicates hallucination risk.
- Verbal uncertainty mismatch (arXiv:2503.14477): the gap between high semantic uncertainty and low verbal uncertainty predicts hallucinations -- LLMs are overconfident when hallucinating.

**What is novel:** No published method runs two attention mechanisms in parallel on the same frozen weights as a dynamical systems probe. SAD combines known ingredients (linear attention, cosine divergence, delay-coordinate embedding via ordinal patterns) in a new configuration. The Takens framing -- treating per-head SAD as an attractor reconstruction rather than a scalar diagnostic -- is the theoretical contribution.

**What is not yet proven:** That the reconstructed attractors carry information about inference regimes rather than reflecting other sources of variation (prompt complexity, sequence length, topic domain). Both the 40-sample pilot and the 400-sample replication are closed: the directional asymmetry observed in the pilot did not survive length-matched permutation testing (p=0.96). Gate 3 tests the central instrument-validation question -- whether per-head PE tracks the fractal dimension of the belief state attractor predicted by Shai et al. (NeurIPS 2024) -- with synthetic HMM benchmarks where ground-truth fractal dimensions are known.

## Current State

- **Milestones A + B:** Complete. Core math, types, I/O, mock hooks, temporal analysis.
- **Milestone C:** Complete. Real instrumentation proven on Mistral-7B.
- **Milestone D (Gates 2-3):** Gate 2 passes. 40-sample pilot and 400-sample replication closed as case studies; recurrence count statistic dead, dense-d direction killed by length-matched null. Full Gate 3 redesigned around synthetic HMM benchmarks with known fractal dimensions (see below).

249 tests (237 CPU + 12 GPU). CI enforces lint, format, typecheck, and test on every PR.

### Instrument Validation Summary

All validation performed on Mistral-7B-Instruct-v0.2 (fp16, eager attention, revision-pinned).

**Gate 0 -- Non-interference.** The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

**Gate 1 -- Parity.** Recomputed fp32 softmax attention, passed through the model's native o_proj, matches the native module output for the newest token. Calibrated across 2240 parity records (32 layers, short + medium sequences, 3 prompt shapes). Frozen thresholds: cosine similarity >= 0.999996 (worst observed: 0.99999869), relative L2 <= 0.002759 (worst observed: 0.00184). Pre-o_proj diagnostic confirms the error source is the expected fp32/fp16 precision asymmetry in the V matmul, not capture or projection bugs.

**Gate 2 -- Stability.** 50 consecutive generations with full instrumentation and JSONL serialization. Zero VRAM creep (0.0 MiB spread across 50 samples, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB). All 50 raw records round-trip through gzipped JSONL with intact provenance: schema fields, per-step/per-layer bijection, StepRecord type reconstruction, and all per-head cosine deltas finite and in [0, 2]. No graph retention, no memory leaks, no serialization drift.

## Scope (Phase 1)

- Model: Mistral-7B only (other families earn entry after Gate 1)
- Benchmark: Synthetic HMM family with known fractal dimensions (Gate 3). TruthfulQA work is closed as a methodological case study; no further application work is committed.
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
| 3 (pilot) | Per-head PE structure on TruthfulQA (40 + 400 samples) | **Closed case study** — recurrence count dead; dense-d direction killed by length-matched null at 400 samples |
| 3 (full) | Rank correlation of per-head PE with known fractal dimension across synthetic HMM family | Planned |

See [ROADMAP.md](ROADMAP.md) for the full research plan.

## Documentation

Full documentation: [project-navi.github.io/navi-SAD](https://project-navi.github.io/navi-SAD/)

Theory, instrument validation, pilot findings, open problems, glossary, and module reference.

## License

Apache-2.0. Copyright Project Navi LLC.
