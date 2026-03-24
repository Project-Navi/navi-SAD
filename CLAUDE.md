# navi-SAD — Claude Code Instructions

## What This Is

Spectral Attention Divergence (SAD): a confabulation detection method. Runs softmax and linear attention in parallel on the same weights, measures per-head cosine divergence, tracks temporal dynamics via ordinal patterns (permutation entropy).

**This is a research harness, not a product.**

## Current State (2026-03-24)

Milestones A + B complete. 63 tests passing. Pure math, types, I/O, registry (Mistral only), and mock hook manager are built and committed.

### What exists and works
- `core/spectral.py` — softmax + linear attention (newest-token), GQA expansion, cosine distance
- `signal/ordinal.py` — Bandt-Pompe ordinal patterns with tie exclusion, permutation entropy (D=3)
- `signal/derivatives.py` — finite differences on delta series
- `signal/aggregation.py` — uniform-mean aggregation from per-layer-per-head to per-token
- `io/writer.py`, `io/reader.py`, `io/derived.py` — raw/derived gzipped JSONL split
- `core/registry.py` — Mistral-only model registry
- `core/hooks.py` — mock hook manager with non-interference proof

### What does NOT exist yet (Milestones C + D)
- Real Mistral family adapter (post-RoPE Q/K/V capture inside native attention forward)
- Gate 0: Non-interference on real model
- Gate 1: Parity validation (recomputed softmax vs native output via o_proj)
- Gate 2: Memory stability under all-layer hooking
- Gate 3: Head sparsity analysis on TruthfulQA
- TruthfulQA benchmark runner + scorer
- Analysis module (AUROC, Cohen's d, bootstrap CIs)

## Plans (local only, gitignored)

- `docs/plans/SPEC.md` — Full instrument design spec (705 lines). Covers method definition, precision strategy, hook architecture, capture tiers, verification gates, phasing.
- `docs/plans/PLAN.md` — Implementation plan v2 (trimmed). Tasks 1-8 done. Tasks 9-11 remain.

**Read SPEC.md before touching anything in Milestone C.** It contains critical contracts:
- Step accounting (one forward = one step, how step_idx increments)
- Tie exclusion policy (tied ordinal windows excluded, not encoded)
- Scorer versioning (truthfulqa_exact_v1 is dev-only)
- Capture tiers (A/B/C) and parity discipline

## Frozen Decisions

| Decision | Choice |
|----------|--------|
| KV cache | **Off** (method definition) |
| Quantization | **q8 minimum, fp16 only for gates** |
| Precision | **Native dtype inference, fp32 instrument branch** |
| Capture boundary | **Post-RoPE Q/K/V** (preferred), hidden-state fallback is Tier C |
| Temporal features | **PE (ordinal, primary) + raw finite differences (supplementary)** |
| Registry scope | **Mistral only** until Gate 1 passes |
| Benchmarks | **TruthfulQA generation** only until Gate 3 passes |
| Baselines | **None** until signal validated across architectures |

## What Comes Next

### Milestone C: Real Instrumentation (fresh session, high attention)

This is the hard part. The mock adapter (hooks.py) proves the plumbing works on FakeAttention. The real adapter must:

1. Patch inside Mistral's native attention forward to capture post-RoPE Q/K/V
2. Read the actual HuggingFace MistralAttention source to find the right tensor locus
3. Preserve exact forward signature, return type, kwargs, device/dtype behavior
4. Pass Gate 0 (identical tokens under deterministic decoding with hooks installed)
5. Pass Gate 1 (recomputed fp32 softmax through native o_proj matches native output)

**Gate 1 tolerance discipline:** Run one calibration pass first. Freeze tolerances. Never relax them after seeing task results.

### Milestone D: First Data

6. TruthfulQA generation runner with chat-template-aware prompting (apply_chat_template per model)
7. Scorer contract: exact match (dev-only scorer_version=truthfulqa_exact_v1)
8. Manual inspection of 25-50 samples before trusting the harness
9. Gate 2: Memory stability (50 long-form generations, VRAM/CPU RAM stable)
10. Gate 3: Head sparsity (200 TruthfulQA samples, per-head Cohen's d, PE as primary discriminator)
11. Analysis module: discrimination.py (AUROC, Cohen's d, bootstrap CIs)

## Code Standards

- `make test` runs offline tests (no GPU)
- `make test-gpu` runs gate tests (requires model)
- `ruff check src/` must pass before commit
- TDD: write tests first, verify failure, implement, verify pass
- Stage specific files — never `git add .` or `git add -A`
- Commits: `<type>: <description>` (feat, fix, refactor, test, chore, docs)
- Raw records are immutable. Derived records are re-generable. No analysis during inference.

## Reference Codebases

- `/home/ndspence/GitHub/navi-donkey/` — Original SAD prototype. `src/donkey/spectral.py` has the validated linear attention core.
- `/home/ndspence/GitHub/projectnavi.ai/project-navi-api/kernel/include/navi_dsc_renyi.h` — Production ordinal patterns + Renyi entropy (C++/WASM). The ordinal.py port references this.
- `/home/ndspence/GitHub/projectnavi.ai/project-navi-api/kernel/include/navi_psi_metrics.h` — PSI temporal derivatives. Deferred for SAD Phase 1 but available as an ablation.
