# navi-SAD Roadmap

## Immediate (no GPU, existing pilot data)

1. **D-sweep on pilot data.** Sweep embedding dimension D from 3 to 7 on existing 40-sample records. Check whether PE stabilizes as D increases. If it stabilizes, embedding dimension is sufficient for the attractor. If it keeps climbing, we're undersampling fractal structure.

2. **Layer-stratified PE profiles.** Plot per-layer PE separation (correct vs incorrect) from L0 to L31 in the pilot data. Tests the progressive construction prediction from Shai et al. (NeurIPS 2024): if later layers have more fully constructed belief state geometry, correct/incorrect separation should grow with depth.

3. **Observable genericity argument.** Write the explicit justification that per-head SAD divergence is a generic observable of the belief state, or state precisely what additional assumption is required. The chain: residual stream encodes belief state → attention is a deterministic function of it → cosine divergence is smooth → SAD delta is h(x_t) where h is smooth. Genericity must be tested empirically via embedding continuity and false-nearest-neighbor statistics.

4. **Flower graph / belief state correspondence.** The mathematical work connecting fd-formalization's (u,v)-flower construction to non-unifilar HMM branching structure. This closes the formal loop from existence theory through fractal geometry to empirical observation.

## Near-term (instrument hardening)

5. **Polish pass.** PerStepDict boundary type, fail-closed fixes, type annotations, named constants, CI coverage for scripts/, test gaps, CLAUDE.md refresh. Spec and plan complete (`docs/plans/POLISH_PASS_SPEC.md`, `docs/plans/POLISH_PASS_PLAN.md`).

6. **Renyi fingerprint.** Port Renyi entropy parameter sweeps (q ∈ [0.1, 7.0]) and Renyi complexity (Jensen-Renyi divergence) from the production C++ kernel (`navi_dsc_renyi.h`). Shannon PE becomes the q=1 special case. The full (H_q, C_q) fingerprint curve per head characterizes the *shape* of the ordinal pattern distribution — the attractor's ordinal signature.

7. **Permutation null test.** Stratified permutation null on the recurrence statistic with eligibility accounting. Analysis module under `src/navi_sad/analysis/` (eligibility, recurrence, permutation). Spec and plan complete (`docs/plans/PE_RECURRENCE_NULL_PLAN.md`).

## Gate 3 (synthetic benchmark)

8. **Synthetic HMM benchmark harness.** Build a family of binary HMMs with known unifilarity properties, ranging from fully unifilar (point attractor) to maximally non-unifilar (known fractal dimension). Generate matched-length sequences, feed to Mistral with next-token prediction framing, capture per-head SAD trajectories.

9. **Gate 3: rank correlation.** Per-head PE (across D=3..7) correlated with known fractal dimension of the generating process. Pass criterion: significant Spearman rank correlation in L15-21 heads, surviving permutation null. This replaces the old TruthfulQA AUROC criterion.

## Post-validation (natural language)

10. **Natural language benchmarks.** Apply validated instrument to controlled natural-language tasks: nested syntactic dependencies, multi-step reasoning with hidden variables, ambiguous pronoun resolution. Only after instrument is validated on known synthetic processes.

11. **TruthfulQA revisited.** Reframe with the new hypothesis: not correctness separation, but belief-state complexity characterization. The instrument measures attractor geometry; TruthfulQA provides one regime partition among many.

## Theoretical integration

12. **Three-repo unification.** cd-formalization (existence theory), fd-formalization (fractal dimension theory), navi-SAD (empirical reconstruction). One instrument suite on one state space. The Shai et al. result provides the empirical anchor: transformers construct belief state geometry in their residual streams, and that geometry can be genuinely fractal.
