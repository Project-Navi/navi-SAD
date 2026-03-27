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

<!-- Phase 2: Full methodology, calibration discipline, tolerance freezing protocol -->
