# navi-SAD -- Claude Code Instructions

## What This Is

Spectral Attention Divergence (SAD): a confabulation detection method. Runs softmax and linear attention in parallel on the same weights, measures per-head cosine divergence, tracks temporal dynamics via ordinal patterns (permutation entropy).

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Current State (2026-03-24)

Milestones A + B complete. Phases C1-C3 merged (PR #6). Gate 0 passes on Mistral-7B. 97 tests (91 CPU + 6 GPU).

### What exists and works
- `core/spectral.py` -- softmax + linear attention (newest-token), GQA expansion, cosine distance
- `core/hooks.py` -- mock hook manager with GQA-correct reshape, non-interference proof
- `core/adapter.py` -- MistralAdapter: Tier A forward-replacement with 2 capture insertions. Verbatim upstream copy from transformers 4.57.x. Runtime version guard. Eager-only hard fail.
- `core/instrument.py` -- InstrumentManager: orchestrates adapters via registry factory, step accounting via LogitsProcessorList, SAD delta computation. No parity yet.
- `core/registry.py` -- Mistral-only model family registry with `adapter_factory`
- `core/types.py` -- StepRecord, RawSampleRecord, ModelFamilyConfig (with adapter_factory)
- `signal/ordinal.py` -- Bandt-Pompe ordinal patterns with tie exclusion, PE (D=3), `recommended_min_pe_length` policy threshold
- `signal/derivatives.py` -- finite differences on delta series
- `signal/aggregation.py` -- uniform-mean aggregation, fail-closed on step_idx gaps
- `io/writer.py`, `io/reader.py`, `io/derived.py` -- raw/derived gzipped JSONL split
- `.github/workflows/ci.yml` -- lint-typecheck + test jobs, uv-first, SHA-pinned, `--locked`
- `.github/dependabot.yml` -- pip + github-actions, torch major blocked
- `tests/gates/test_gate0_noninterference.py` -- Gate 0: token identity, logit exact match, per-step/per-layer bijection. Passes on Mistral-7B-Instruct-v0.2 (fp16, eager, revision-pinned).
- `tests/gates/conftest.py` -- GPU fixture: model load with deterministic CUDA controls, revision-pinned

### What does NOT exist yet

**Phase C4 (next -- parity extension):**
- Insertion 3 (pre-o_proj diagnostic capture in adapter)
- ParityConfig + ParityRecord types
- Parity mode in InstrumentManager
- Gate 1 calibration script + frozen-tolerance Gate 1 pytest

**Milestone D (after C passes):**
- Gate 2: Memory stability under all-layer hooking (50 long-form generations)
- Gate 3: Head sparsity analysis on TruthfulQA (200 samples, per-head Cohen's d)
- TruthfulQA benchmark runner + scorer
- Analysis module (AUROC, Cohen's d, bootstrap CIs)

## Plans (local only, gitignored)

- `docs/plans/SPEC.md` -- Full instrument design spec (~705 lines). Method definition, precision strategy, hook architecture, capture tiers, verification gates.
- `docs/plans/PLAN.md` -- Implementation plan v2 (Milestones A-D). Tasks 1-8 done.
- `docs/plans/MILESTONE_C_PLAN.md` -- Phase C1-C3 implementation plan. Complete.
- `docs/plans/PHASE_C4_SPEC.md` -- Phase C4 design spec. Parity extension + Gate 1.
- `docs/plans/PHASE_C4_PLAN.md` -- Phase C4 implementation plan. Next up.
- `docs/plans/CI_CD_PROPOSAL.md` -- CI/CD research and proposal. Implemented.

**Read SPEC.md before touching anything in Milestone C.** It contains critical contracts:
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
| Registry scope | **Mistral only** until Gate 1 passes |
| Benchmarks | **TruthfulQA generation** only until Gate 3 passes |
| Baselines | **None** until signal validated across architectures |
| Package manager | **uv** exclusively. No pip fallback. Lockfile committed. |
| Transformers | **~=4.57** pinned. Forward-replacement adapter is version-coupled. |
| Attention impl | **eager only** for instrumented models. Hard fail on non-eager. |
| Model revision | **Pinned** in gate fixtures. Update only after re-validating gates. |
| License | **Apache-2.0** (Copyright Project Navi LLC) |

## Milestone C: Real Instrumentation

This is where the project stops being "well-structured code" and starts being "instrument that can lie." The standard is different: fewer files, tighter diffs, more brutal tests.

### Phases C1-C3: Complete (PR #6)
- MistralAdapter: verbatim forward copy from transformers 4.57.x, 3 marked insertions (capture, parity, pre-o_proj diagnostic). Eager-only hard fail. Runtime version guard.
- InstrumentManager: registry-driven adapter creation, step accounting via LogitsProcessorList.
- Gate 0: passes on Mistral-7B-Instruct-v0.2 (fp16, eager, revision-pinned). Token identity, logit exact match, per-step/per-layer bijection.
- transformers pinned to ~=4.57. accelerate added as dependency.

### Phase C4: Next (parity + Gate 1)
- ParityConfig + ParityRecord types
- Parity mode in InstrumentManager (recompute fp32 softmax, through native o_proj, compare)
- Pre-o_proj diagnostic (debug channel, not gated)
- Gate 1 calibration script (one-off, not CI)
- Gate 1 pytest with frozen tolerances (cosine + relative L2 only, max-abs is diagnostic)

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

**Out of scope for Milestone C:** benchmarks, Llama, Phi, scoring, analysis, anything beyond Mistral + Gate 0 + Gate 1.

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
