# navi-SAD -- Claude Code Instructions

## What This Is

Spectral Attention Divergence (SAD): a dynamical systems probe for LLM inference. Runs softmax and linear attention in parallel on the same weights, measures per-head cosine divergence, and reconstructs the model's internal attractor via delay-coordinate embedding (permutation entropy).

**Takens' embedding framing:** Each per-head SAD trajectory is a delay-coordinate embedding of the model's internal dynamical state. We are not measuring a signal — we are reconstructing an attractor. PE is load-bearing: Bandt-Pompe ordinal patterns are designed for delay-coordinate reconstructions. The instrument is hypothesis-agnostic — it reconstructs dynamical structure. Confabulation detection remains one application (attractor collapse correlating with incorrect generation), but the instrument can characterize any regime that leaves a signature in per-head attention dynamics.

**This is a research harness, not a product. The instrument can lie. Every claim requires evidence.**

## Current State (2026-03-25)

Milestone C complete. Gates 0, 1, 2 pass on Mistral-7B. Gate 3 pilot complete (40 samples, manually labeled). PE feature layer built. 249 tests (237 CPU + 12 GPU). See [ROADMAP.md](ROADMAP.md) for research priorities.

### Pilot findings (characterization, not evidential)

The 40-sample pilot falsified the naive hypothesis and produced one result worth pursuing:

- **Grand-mean SAD does not separate groups.** 0.006 gap on ~0.30 baseline. Dead.
- **Per-(layer, head) mean delta has structure.** Leading-span scalar: 294/1024 heads with |d|>0.5. Late layers flip sign.
- **Per-head PE on first-differenced SAD is the strongest signal.** 338/1024 heads show |d|>0.5 across 3+ (mode, segment) combinations. Directional asymmetry: 4.6:1 positive (correct = more complex dynamics, incorrect = more stereotyped). Cross-mode recurrence across raw, diff, and residual.
- **Position confound confirmed.** Both groups climb from ~0.24 to ~0.40 over generation. First-differencing removes the trend but signal persists.
- **Shadow scorer dead.** 10% agreement. Manual labels (3-reviewer majority vote, 92% unanimous) are canonical.

**Hypothesis revised:** SAD is not a truth detector. It is a dynamical systems probe that reconstructs per-head attractor structure. The theoretical anchor is Shai et al. (NeurIPS 2024, arXiv:2405.15943): transformers construct belief state geometry in their residual streams, and that geometry can be genuinely fractal for non-unifilar inference processes. Gate 3 tests whether per-head PE tracks the computational-mechanical complexity of the inference problem, using synthetic HMM benchmarks with known fractal dimensions.

### What exists and works

- `core/spectral.py` -- softmax + linear attention (newest-token), GQA expansion, cosine distance
- `core/hooks.py` -- mock hook manager with GQA-correct reshape, non-interference proof
- `core/adapter.py` -- MistralAdapter: Tier A forward-replacement with 3 marked insertions. Verbatim upstream copy from transformers 4.57.x. Runtime version guard. Eager-only hard fail.
- `core/instrument.py` -- InstrumentManager: orchestrates adapters via registry factory, step accounting via LogitsProcessorList, SAD delta computation.
- `core/registry.py` -- Mistral-only model family registry with `adapter_factory`
- `core/types.py` -- StepRecord, RawSampleRecord, ModelFamilyConfig, ParityConfig, ParityRecord
- `signal/ordinal.py` -- Bandt-Pompe ordinal patterns with tie exclusion, PE (D=3). Generic engine, NOT SAD-specific.
- `signal/pe_features.py` -- SAD-specific PE wrapper: per-(layer, head) extraction, first-differencing, detrending, segmentation, eligibility gating. Does not modify ordinal.py.
- `signal/derivatives.py` -- finite differences on delta series
- `signal/aggregation.py` -- uniform-mean aggregation, fail-closed on step_idx gaps
- `signal/types.py` -- OrdinalResult, DerivedSampleRecord dataclasses
- `pilot/schema.py` -- Typed write-side schema: frozen PilotSampleRecord, PilotReviewRecord, PilotMetadata dataclasses with __post_init__ enum validation. Label, DisagreementCategory, StopReason, SpanStopReason enums. Derived REVIEW_READONLY_FIELDS.
- `pilot/helpers.py` -- extraction, shadow scorer, scalar computation, alignment, integrity validation, guarded Cohen's d, confusion matrix
- `scripts/pilot_gate3.py` -- generation (40-sample TruthfulQA) + `--analyze` entry points. Incremental artifact persistence, invalid-sample flagging, deterministic CUDA controls.
- `io/writer.py`, `io/reader.py`, `io/derived.py` -- raw/derived gzipped JSONL split
- `.github/workflows/ci.yml` -- lint-typecheck + test jobs, uv-first, SHA-pinned, `--locked`
- `.pre-commit-config.yaml` -- local pre-commit hooks mirroring CI (ruff check, ruff format, mypy)
- `AGENTS.md` -- Internal Affairs (Perplexity) auditor prompt
- `CITATION.cff` -- project citation metadata

### Pilot artifacts (gitignored, results/)

- `results/pilot_gate3/samples.json` -- 40 samples with per-step per-layer per-head SAD deltas
- `results/pilot_gate3/review.json` -- 3-reviewer majority-vote labels (28 correct, 9 incorrect, 3 ambiguous)
- `results/pilot_gate3/raw.jsonl.gz` -- raw JSONL archive
- `results/pilot_gate3/cohens_d.json` -- per-(layer, head) Cohen's d matrices (exploratory, not evidential)

### What does NOT exist yet

**Next immediate steps (in order):**
1. D-sweep on pilot data (D=3..4 feasible under 2*D! policy; D=5+ requires longer sequences or relaxed eligibility)
2. Layer-stratified PE profiles (per-layer correct/incorrect separation, L0-L31)
3. Observable genericity argument (justify per-head SAD as generic observable of belief state)
4. Polish pass (PerStepDict boundary type, fail-closed fixes, type annotations, CI coverage)
5. Permutation null test (stratified null on recurrence statistic with eligibility accounting)
6. Rényi fingerprint (port Rényi entropy parameter sweeps from production C++ kernel)

**Milestone D (remaining):**
- Gate 3: synthetic HMM benchmark — rank correlation of per-head PE with known fractal dimension across a family of generating processes with known unifilarity properties
- Analysis module under `src/navi_sad/analysis/` (eligibility, recurrence, permutation)
- TruthfulQA revisited post-validation as one regime partition among many
- Gate 6: Overhead measurement (informational, not blocking)

**Deferred (from IA audit, post-pilot):**
- Adapter AST fingerprint check (IA F-04)
- Linear attention denominator health diagnostics (IA F-11)
- Full position-aware SAD normalization (IA F-09)
- per_step typed record (Grumpy F-01 from PR #17)

## Plans (local only, gitignored)

- `docs/plans/SPEC.md` -- Full instrument design spec (~705 lines).
- `docs/plans/PLAN.md` -- Implementation plan v2 (Milestones A-D).
- `docs/plans/GATE3_PILOT_PLAN.md` -- Gate 3 pilot implementation plan. Implemented (PR #15).
- `docs/plans/GATE3_PILOT_SPEC.md` -- Gate 3 pilot design spec. Implemented. 2 rounds Grumpy + 2 rounds IA.
- `docs/plans/POLISH_PASS_SPEC.md` -- Polish pass spec (PerStepDict, fail-closed, type annotations, CI).
- `docs/plans/POLISH_PASS_PLAN.md` -- Polish pass implementation plan.
- `docs/plans/PE_RECURRENCE_NULL_PLAN.md` -- Permutation null test plan for PE recurrence statistic.
- `docs/audit/IA_RESPONSE_2026-03-25.md` -- Formal IA audit response with dispositions.
- `docs/audit/SESSION_REPORT_2026-03-25.md` -- Full session report with bugs-caught-by-auditors analysis.

**Read SPEC.md before touching anything.** It contains critical contracts:
- Step accounting (Forward 0 = token 1, expected records = num_layers * max_new_tokens)
- Tie exclusion policy (tied ordinal windows excluded, not encoded)
- Capture tiers (A/B/C) and parity discipline
- Gate 1 tolerance discipline (calibrate first, freeze, never relax)

## Frozen Decisions

| Decision | Choice |
|----------|--------|
| KV cache | **Off** (method definition; scope limitation -- generalization to cache-on inference is unverified) |
| Quantization | **q8 minimum, fp16 only for gates** |
| Precision | **Native dtype inference, fp32 instrument branch** |
| Capture boundary | **Post-RoPE Q/K/V** (preferred), hidden-state fallback is Tier C |
| Temporal features | **PE per-(layer, head) on first-differenced SAD trajectories (primary) + raw finite differences (supplementary) + Rényi fingerprint (planned)**. PE on pooled grand means is dead. |
| Registry scope | **Mistral only** until cross-family gates pass |
| Benchmarks | **Synthetic HMM sequences** for Gate 3 (rank correlation with known fractal dimension). TruthfulQA deferred to post-validation. |
| Baselines | **None** until signal validated across architectures |
| Package manager | **uv** exclusively. No pip fallback. Lockfile committed. |
| Transformers | **~=4.57** pinned. Forward-replacement adapter is version-coupled. |
| Attention impl | **eager only** for instrumented models. Hard fail on non-eager. |
| Model revision | **Pinned** in gate fixtures. Update only after re-validating gates. |
| Dataset revision | **Pinned** in pilot script (`741b8276...`). |
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
- Gate 1 calibration: 2240 records. Frozen thresholds: cosine >= 0.999996, relative L2 <= 0.002759.
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

### Analysis discipline
- Analysis code is part of the instrument. It lives in `src/navi_sad/`, not in hand-written scripts.
- No hand-written analysis scripts producing results we claim to trust. If we don't have what we need, build it properly, verify it, then consider trusting the results.
- Scripts in `scripts/` are entry points that call tested modules, not analysis logic.
- Statistical computations (permutation nulls, effect sizes, eligibility) require the same TDD, type annotation, and edge-case coverage as instrument code.
- A bug in analysis code is as dangerous as a bug in the adapter — it can silently produce wrong p-values.

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
- Pre-commit hooks mirror CI: `uv run pre-commit install`

### Docs
The docs site uses [Zensical](https://zensical.org/) with Project Navi branding.

```bash
# Install docs dependencies
uv sync --no-install-project --group docs

# Local preview
uv run zensical serve    # or: make docs-serve

# Build static site
uv run zensical build    # or: make docs-build
```

- `zensical.toml` -- site config (edit `[project]` table and `nav`)
- `docs/` -- all published markdown content
- `docs/stylesheets/navi.css` -- brand CSS (shared, don't edit per-project)
- `.github/workflows/docs.yml` -- CI: build and deploy to GitHub Pages

### Quality
- Ruff rules: E, F, I, W, B (bugbear), UP (pyupgrade), RUF, C4 (comprehensions)
- Ruff pinned: `~=0.15.0`. Mypy pinned: `~=1.19.0`. Local and CI must match.
- Mypy: `check_untyped_defs = true`, per-module `ignore_missing_imports`, `show_error_codes`
- Stage specific files -- never `git add .` or `git add -A`
- Commits: `<type>: <description>` (feat, fix, refactor, test, chore, docs)
- PRs require passing CI (`lint-typecheck` + `test`) and 1 approval
- Linear history enforced on main (squash-merge or rebase only)

### Data integrity
- Raw records are immutable. Derived records are re-generable. No analysis during inference.
- `aggregate_deltas()` raises on non-contiguous step_idx. No silent zero-fill.
- `compute_mean_delta_matrix()` raises on missing layers. No silent zero-fill.
- PE eligibility is a configurable policy threshold (`recommended_min_pe_length`), not a mathematical law. SAD-specific wrapper uses 2*D! minimum (stricter).

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
