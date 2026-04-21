---
name: gate-check
description: Run Gate 0/1/2 verification against frozen tolerances for the SAD instrument. User-only — never auto-invoke. Use before releasing changes to core/adapter.py, core/instrument.py, after bumping transformers, or before a trusted-output analysis run.
disable-model-invocation: true
---

# Gate Check

Verify the SAD instrument against its frozen gates.

## Why this exists

Every claim in navi-SAD rests on three gates passing with frozen tolerances:

- **Gate 0** (non-interference) — hook installation must not change model output. Exact token identity + exact logit equality.
- **Gate 1** (parity) — instrument-branch softmax must match native attention per-head within frozen cosine / relative-L2 thresholds.
- **Gate 2** (stability) — 50 consecutive generations with full instrumentation: zero VRAM creep, bounded CPU growth, provenance round-trip.

Tolerances are **calibrated once, frozen, and never relaxed after observing results** (CLAUDE.md § "Gate 1 tolerance discipline"). A gate that "almost passes" has failed. Fix the instrument, not the threshold.

## When to run

- Before merging any change to `src/navi_sad/core/adapter.py` or `src/navi_sad/core/instrument.py`.
- After any bump to `transformers` in `pyproject.toml` (the patched forward is version-coupled: `4.57.0 ≤ version < 4.58.0`).
- After pinning a new Mistral model revision in the gate fixtures.
- Before a batch analysis run where trusted-output integrity matters.

## Prerequisites

- GPU available, Mistral weights downloaded at the pinned revision.
- Clean `uv sync --extra dev --locked`.

## Procedure

1. Run the gates:

   ```bash
   make test-gpu
   ```

   This executes `tests/gates/test_gate0_noninterference.py`, `test_gate1_parity.py`, and `test_gate2_stability.py` via the `@pytest.mark.gpu` marker.

2. On any failure, do NOT relax a threshold. Instead:
   - Diff against the previous green commit.
   - Check the transformers version (`uv run python -c "import transformers; print(transformers.__version__)"`) against `_COMPAT_MIN` / `_COMPAT_MAX` in `core/adapter.py`.
   - If adapter code changed, run the `adapter-upstream-diff` skill to localise drift.
   - Re-run Gate 1 calibration only if the transformers version legitimately bumped: `uv run python scripts/calibrate_gate1.py`. Freeze the new thresholds in a dedicated PR that calls out the calibration explicitly.

3. If all gates pass, record the commit SHA and the transformers / revision versions in the PR description.

## Failure modes to watch for

- **Gate 0 token mismatch** — hook or adapter is mutating native output. Most likely cause: an in-place op inside an INSERTION block. Do not proceed until tokens match exactly.
- **Gate 1 cosine < 0.999996 or relative L2 > 0.002759** — parity-branch divergence from native attention. Inspect the pre-o_proj diagnostic (INSERTION 3) output to localise which layer / head is drifting.
- **Gate 2 VRAM creep > 16 MiB** — hooks leaking references. Do not merge.

## Why this is user-only

`disable-model-invocation: true` is deliberate: gate verification is too consequential to be auto-invoked on every adapter touch. The operator runs it explicitly, reads the output, and decides whether to proceed.

## References

- `CLAUDE.md` § "Phases C1-C3", "Phase C4", "Gate 2", "Gate 1 tolerance discipline"
- `tests/gates/` — current test implementations
- `scripts/calibrate_gate1.py` — calibration entrypoint
