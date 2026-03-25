# navi-SAD -- Claude Code Instructions

## What This Is

Spectral Attention Divergence (SAD): a confabulation detection method. Runs softmax and linear attention in parallel on the same weights, measures per-head cosine divergence, tracks temporal dynamics via ordinal patterns (permutation entropy).

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Current State (2026-03-25)

Milestone C complete. Gates 0, 1, 2 pass on Mistral-7B. 119 tests (107 CPU + 12 GPU).

### What exists and works
- `core/spectral.py` -- softmax + linear attention (newest-token), GQA expansion, cosine distance
- `core/hooks.py` -- mock hook manager with GQA-correct reshape, non-interference proof
- `core/adapter.py` -- MistralAdapter: Tier A forward-replacement with 3 marked insertions (capture, pre-o_proj diagnostic, parity). Verbatim upstream copy from transformers 4.57.x. Runtime version guard. Eager-only hard fail.
- `core/instrument.py` -- InstrumentManager: orchestrates adapters via registry factory, step accounting via LogitsProcessorList, SAD delta computation, parity mode with ParityConfig.
- `core/registry.py` -- Mistral-only model family registry with `adapter_factory`
- `core/types.py` -- StepRecord, RawSampleRecord, ModelFamilyConfig, ParityConfig, ParityRecord
- `signal/types.py` -- OrdinalResult, DerivedSampleRecord dataclasses
- `signal/ordinal.py` -- Bandt-Pompe ordinal patterns with tie exclusion, PE (D=3), `recommended_min_pe_length` policy threshold
- `signal/derivatives.py` -- finite differences on delta series
- `signal/aggregation.py` -- uniform-mean aggregation, fail-closed on step_idx gaps
- `io/writer.py`, `io/reader.py`, `io/derived.py` -- raw/derived gzipped JSONL split
- `.github/workflows/ci.yml` -- lint-typecheck + test jobs, uv-first, SHA-pinned, `--locked`
- `.github/dependabot.yml` -- pip + github-actions, torch major blocked
- `scripts/calibrate_gate1.py` -- one-off Gate 1 calibration (not CI)
- `tests/gates/conftest.py` -- GPU fixture: model load with deterministic CUDA controls, revision-pinned
- `tests/gates/test_gate0_noninterference.py` -- Gate 0: token identity, logit exact match, per-step/per-layer bijection
- `tests/gates/test_gate1_parity.py` -- Gate 1: frozen cosine + relative L2 thresholds, layer drift invariant
- `tests/gates/test_gate2_stability.py` -- Gate 2: 50 generations, VRAM stability, provenance round-trip

### What does NOT exist yet

**Milestone D (next):**
- Gate 3: Head sparsity analysis on TruthfulQA (200 samples, per-head Cohen's d)
- TruthfulQA benchmark runner + scorer
- Analysis module (AUROC, Cohen's d, bootstrap CIs)
- Gate 6: Overhead measurement (informational, not blocking)

## Plans (local only, gitignored)

- `docs/plans/SPEC.md` -- Full instrument design spec (~705 lines). Method definition, precision strategy, hook architecture, capture tiers, verification gates.
- `docs/plans/PLAN.md` -- Implementation plan v2 (Milestones A-D). Milestones A-C tasks complete. Milestone D tasks remain.
- `docs/plans/MILESTONE_C_PLAN.md` -- Phase C1-C3 implementation plan. Complete.
- `docs/plans/PHASE_C4_SPEC.md` -- Phase C4 design spec. Parity extension + Gate 1. Complete.
- `docs/plans/PHASE_C4_PLAN.md` -- Phase C4 implementation plan. Complete.
- `docs/plans/GATE2_SPEC.md` -- Gate 2 design spec. Stability + serialization. Complete.
- `docs/plans/GATE2_PLAN.md` -- Gate 2 implementation plan. Complete.
- `docs/plans/GATE3_PILOT_PLAN.md` -- Gate 3 pilot implementation plan. Pending implementation.
- `docs/plans/GATE3_PILOT_SPEC.md` -- Gate 3 pilot design spec. Approved, pending implementation.
- `docs/plans/CI_CD_PROPOSAL.md` -- CI/CD research and proposal. Implemented.

**Read SPEC.md before touching anything.** It contains critical contracts:
- Step accounting (Forward 0 = token 1, expected records = num_layers * max_new_tokens)
- Tie exclusion policy (tied ordinal windows excluded, not encoded)
- Capture tiers (A/B/C) and parity discipline
- Gate 1 tolerance discipline (calibrate first, freeze, never relax)

## Frozen Decisions

| Decision | Choice |
|----------|--------|
| KV cache | **Off** (method definition) |
| Quantization | **q8 minimum, fp16 only for gates** |
| Precision | **Native dtype inference, fp32 instrument branch** |
| Capture boundary | **Post-RoPE Q/K/V** (preferred), hidden-state fallback is Tier C |
| Temporal features | **PE (ordinal, primary) + raw finite differences (supplementary)** |
| Registry scope | **Mistral only** until cross-family gates pass |
| Benchmarks | **TruthfulQA generation** only until Gate 3 passes |
| Baselines | **None** until signal validated across architectures |
| Package manager | **uv** exclusively. No pip fallback. Lockfile committed. |
| Transformers | **~=4.57** pinned. Forward-replacement adapter is version-coupled. |
| Attention impl | **eager only** for instrumented models. Hard fail on non-eager. |
| Model revision | **Pinned** in gate fixtures. Update only after re-validating gates. |
| License | **Apache-2.0** (Copyright Project Navi LLC) |

## Milestone C: Real Instrumentation (Complete)

This is where the project stopped being "well-structured code" and started being "instrument that can lie."

### Phases C1-C3 (PR #6)
- MistralAdapter: verbatim forward copy from transformers 4.57.x, 3 marked insertions (capture, pre-o_proj diagnostic, parity). Eager-only hard fail. Runtime version guard.
- InstrumentManager: registry-driven adapter creation, step accounting via LogitsProcessorList.
- Gate 0: passes on Mistral-7B-Instruct-v0.2 (fp16, eager, revision-pinned). Token identity, logit exact match, per-step/per-layer bijection.

### Phase C4 (PR #9)
- ParityConfig + ParityRecord types. Parity mode in InstrumentManager.
- Recomputes fp32 softmax, transpose-then-reshape head merge, downcast through native o_proj, compare against native output newest-token slice. All metrics in fp32.
- Pre-o_proj diagnostic: captures native attention output before projection for failure localization.
- Gate 1 calibration: 2240 records. Frozen thresholds: cosine >= 0.999996, relative L2 <= 0.00276.
- Gate 1 passes. Layer drift invariant holds.

### Gate 2 (PR #11)
- 50 consecutive generations with full instrumentation and JSONL serialization.
- Zero VRAM creep (0.0 MiB spread, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB).
- Provenance round-trip: schema fields, per-step/per-layer bijection, StepRecord reconstruction, delta range validation.

### Adapter discipline
- The patched forward is a VERBATIM COPY of upstream. Not a reimplementation.
- Three marked insertion points: capture_fn after RoPE, pre_oproj diagnostic before reshape, parity_fn after o_proj.
- No refactoring of upstream code. Any change requires Gate 0 re-verification.
- `attn_implementation="eager"` required. Non-negotiable.

### Gate 1 tolerance discipline
- Run one calibration pass first (`scripts/calibrate_gate1.py`). Freeze tolerances. Never relax after seeing task results.
- Gate metrics: cosine similarity + relative L2 (frozen thresholds on every record).
- Diagnostics only: max absolute error, pre-o_proj cosine, layer drift.
- Relative L2 formula: `||recomputed - native||_2 / (||native||_2 + 1e-12)`, all comparisons in fp32.

### Gate 2 stability discipline
- Fixed thresholds (16 MiB VRAM, 128 MiB CPU RSS), not calibrated from observed behavior.
- 2 warmup runs + max-of-first-3 baseline. If early samples are unstable, increase warmup -- never the threshold.
- Provenance validated with per-step/per-layer bijection, StepRecord type reconstruction, and per_head_delta range check.

## Verification Standards

These are not guidelines. They are requirements.

### Gate discipline
- Every gate has a pass/fail criterion defined before implementation
- Gate tolerances are calibrated once, frozen, and never relaxed after observing results
- A gate that "almost passes" has failed. Fix the instrument, not the threshold.
- Gate tests are tagged `@pytest.mark.gpu` and run via `make test-gpu`

### Non-interference invariant
- Hook installation must not change model output under any condition
- "Close enough" is not identical. Use exact tensor equality for deterministic paths.
- For non-deterministic paths, define tolerance before measuring, not after.

### Precision discipline
- Native dtype for inference. fp32 for the instrument branch.
- No silent dtype coercion. Assert dtypes at capture boundaries.
- If a tensor is supposed to be fp32, check that it is fp32.

### Test standards
- TDD: write the failing test first, then implement
- Every behavioral claim in a docstring must have a test that proves it
- Mock tests prove plumbing. Gate tests prove the instrument. Do not confuse them.
- Do not trust test output without running the tests. AI code has higher bug rates.

## Code Standards

### Tooling
- `make all` runs lint + format + typecheck + test (must pass before commit)
- `make test` runs offline CPU tests (no GPU required)
- `make test-gpu` runs gate tests (requires model)
- CI enforces: `uv run ruff check`, `uv run ruff format --check`, `uv run mypy`, `uv run pytest`
- CI uses `uv sync --extra dev --locked` -- stale lockfile fails the build

### Quality
- Ruff rules: E, F, I, W, B (bugbear), UP (pyupgrade), RUF, C4 (comprehensions)
- Mypy: `check_untyped_defs = true`, per-module `ignore_missing_imports`, `show_error_codes`
- Stage specific files -- never `git add .` or `git add -A`
- Commits: `<type>: <description>` (feat, fix, refactor, test, chore, docs)
- PRs require passing CI (`lint-typecheck` + `test`) and 1 approval
- Linear history enforced on main (squash-merge or rebase only)

### Data integrity
- Raw records are immutable. Derived records are re-generable. No analysis during inference.
- `aggregate_deltas()` raises on non-contiguous step_idx. No silent zero-fill.
- PE eligibility is a configurable policy threshold (`recommended_min_pe_length`), not a mathematical law.

### Scope discipline
- Do not add features, refactoring, or "improvements" beyond what was asked
- Do not create abstractions for one-time operations
- If replacing code, delete the old version in the same commit
- Every check and every file must earn its place. No placeholders. No ceremony.
- Security-critical paths (if any) require tests before merge

## Reference Codebases

- `/home/ndspence/GitHub/navi-donkey/` -- Original SAD prototype. `src/donkey/spectral.py` has the validated linear attention core.
- `/home/ndspence/GitHub/projectnavi.ai/project-navi-api/kernel/include/navi_dsc_renyi.h` -- Production ordinal patterns + Renyi entropy (C++/WASM). The ordinal.py port references this.
- `/home/ndspence/GitHub/projectnavi.ai/project-navi-api/kernel/include/navi_psi_metrics.h` -- PSI temporal derivatives. Deferred for SAD Phase 1 but available as an ablation.
