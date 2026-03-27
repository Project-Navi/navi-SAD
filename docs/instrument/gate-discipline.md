# Gate Discipline

*Status: Proven by gates.*

Every gate has a pass/fail criterion defined before implementation. Gate tolerances are calibrated once, frozen, and never relaxed after observing results. A gate that "almost passes" has failed. Fix the instrument, not the threshold.

This prevents the most common form of measurement self-deception: adjusting the instrument until it confirms the hypothesis.

| Gate | Claim | Test | Evidence | Status |
|------|-------|------|----------|--------|
| [0](gate-0.md) | Observer does not perturb the system | Bit-identical tokens and logits | Greedy decode, 32 layers | **Pass** |
| [1](gate-1.md) | Recomputed attention matches native output | Cosine >= 0.999996, rel L2 <= 0.002759 | 2240 records, frozen thresholds | **Pass** |
| [2](gate-2.md) | No memory leaks under sustained use | 50 generations | 0.0 MiB VRAM spread, 0.7 MiB CPU RSS growth | **Pass** |
| 3 | PE tracks belief state complexity | Rank correlation with known fractal dimension | Planned | **In progress** |

---

## Calibration discipline

Gate tolerances follow a three-step protocol:

1. **Define pass/fail criteria before implementation.** Every gate has a documented criterion in the design spec before any code is written. The criterion is the hypothesis; the test either confirms or falsifies it.

2. **Calibrate once, then freeze.** For Gate 1, a single calibration pass runs the adapter across all layers and sequence positions on Mistral-7B-Instruct-v0.2. The worst-case metrics from that pass set the thresholds. Calibration produced 2240 parity records. The observed worst-case cosine delta from 1.0 was 1.31e-6; the frozen threshold applies 3x headroom (cosine >= 0.999996). The observed worst-case relative L2 was 0.00184; the frozen threshold is 0.002759 (1.5x headroom). These numbers live in `tests/gates/test_gate1_parity.py` as module-level constants.

3. **Never relax after seeing task results.** If a gate fails against frozen tolerances, the adapter is broken. Fix the adapter or downgrade the capture tier. Adjusting the threshold to make results pass is the single most dangerous thing an instrument builder can do.

## Gate metrics vs. diagnostics

Not every number produced during validation is a gate metric. The distinction matters:

**Gate metrics** have frozen thresholds. Every record must pass. A single violation fails the gate.

| Metric | Gate | Threshold |
|--------|------|-----------|
| Cosine similarity | 1 | >= 0.999996 |
| Relative L2 error | 1 | <= 0.002759 |
| Token identity | 0 | Exact (`torch.equal`) |
| Logit identity | 0 | Exact (`atol=0, rtol=0`) |
| VRAM spread | 2 | <= 16 MiB |
| CPU RSS growth | 2 | <= 128 MiB |

Relative L2 formula: \( \frac{\lVert \text{recomputed} - \text{native} \rVert_2}{\lVert \text{native} \rVert_2 + 10^{-12}} \), all comparisons in fp32.

**Diagnostics** are reported but do not gate. They exist for failure localization:

- **Max absolute error** (Gate 1): per-record, not thresholded.
- **Pre-o_proj cosine** (Gate 1): isolates whether a parity failure originates in the attention computation or in the head-merge/projection path.
- **Layer drift** (Gate 1): per-layer mean cosine across steps. Error should not systematically grow with depth.

## Fail-closed semantics

"Fail-closed" means: when the instrument detects an anomaly, it stops. It does not interpolate, impute, or degrade gracefully.

- `aggregate_deltas()` raises on non-contiguous `step_idx`. No silent zero-fill.
- `compute_mean_delta_matrix()` raises on missing layers. No silent zero-fill.
- Gate 0 failure halts all downstream work. If the observer perturbs the system, every measurement is suspect.
- Gate tests are tagged `@pytest.mark.gpu` and run via `make test-gpu`. They require the actual model on actual hardware. Mock tests prove plumbing; gate tests prove the instrument.
