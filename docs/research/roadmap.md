# Roadmap

*Last updated: 2026-03-27*

*Status: Active.*

---

This is not a feature roadmap. It is a research plan --- each step either validates or falsifies something about the instrument and the theory behind it.

## Where we have been

### Milestones A--C: Building an instrument that can lie

The first three milestones built the core SAD instrument and proved it works mechanically.

**Milestone A** built the spectral core: softmax and linear attention running in parallel on the same weights, cosine divergence measured per head. The linear attention implementation (newest-token normalization, GQA expansion) was ported from the validated navi-donkey prototype.

**Milestone B** built the signal layer: Bandt-Pompe ordinal patterns with tie exclusion, [permutation entropy](../theory/takens-embedding.md) at D=3, finite differences for temporal derivatives, and fail-closed aggregation that raises on step-index gaps rather than silently zero-filling.

**Milestone C** made the instrument real. The MistralAdapter is a verbatim copy of the upstream transformers 4.57.x forward pass with three marked insertions. [Gate 0](../instrument/gate-discipline.md) proved non-interference (bit-identical tokens and logits). [Gate 1](../instrument/gate-discipline.md) proved parity (cosine >= 0.999996 on 2240 records, thresholds calibrated once and frozen). [Gate 2](../instrument/gate-discipline.md) proved stability (zero VRAM creep over 50 consecutive generations). At this point the instrument was verified to not corrupt inference and to produce consistent measurements. It was not yet verified to measure anything meaningful.

### The pilot: killing a hypothesis

The [40-sample TruthfulQA pilot](pilot-findings.md) killed the naive hypothesis --- grand-mean SAD does not separate correct from incorrect generations (0.006 gap on ~0.30 baseline). But it surfaced per-head PE structure: 338/1024 heads with |d|>0.5 on first-differenced trajectories, a 4.6:1 directional asymmetry (correct = more complex dynamics), and cross-mode recurrence. The PE feature layer (`pe_features.py`) was built to extract this signal properly: per-(layer, head), first-differencing, detrending, segmentation, eligibility gating.

This is where the project changed direction. SAD is not a truth detector. It is a [dynamical systems probe](../theory/sad-instrument.md) that reconstructs per-head attractor structure via [Takens' delay-coordinate embedding](../theory/takens-embedding.md).

## Where we are

Four urgent priorities, all executable on existing pilot data without new GPU runs, followed by instrument hardening and the redesigned Gate 3.

### Priority 1: D-sweep on pilot data

*No GPU needed. Existing artifacts.*

Sweep embedding dimension D from 3 to 4 on the 40-sample pilot records. Under the 2*D! eligibility policy, D=3 requires 12-step minimum windows and D=4 requires 48-step. The pilot trajectories are ~77 steps, so D=4 is feasible. D=5 requires 240-step windows and is infeasible on pilot data --- if needed, it becomes a requirement for the Gate 3 benchmark design.

**Why it matters:** If PE changes substantially between D=3 and D=4, the embedding dimension matters and D=3 may be undersampling the attractor. If PE stabilizes, D=3 is sufficient and the pilot findings hold. This is a direct test of whether the delay-coordinate reconstruction has enough dimensions to capture the attractor's topology.

### Priority 2: Layer-stratified PE profiles

*No GPU needed. Existing artifacts.*

Plot per-layer PE separation (correct vs incorrect) from L0 to L31. This tests the progressive construction prediction from Shai et al. (NeurIPS 2024): if later layers have more fully constructed belief state geometry, correct/incorrect separation should grow with depth.

**Why it matters:** If separation grows from early to late layers, it is convergent evidence that SAD observes belief state geometry being constructed layer by layer. If separation concentrates in L15--21, it matches both the pilot's recurring-head zone and the progressive construction prediction. If separation is flat, the progressive construction story does not apply to this observable.

### Priority 3: Observable genericity argument

*Theoretical work.*

Write the explicit justification that per-head SAD divergence is a generic observable of the belief state, or state precisely what additional assumption is required.

The argument chain: residual stream state x_t encodes belief state (Shai et al., empirical) --- attention computation is a deterministic function of x_t --- cosine divergence between softmax and linear attention outputs is a composition of smooth functions --- therefore SAD delta is h(x_t) where h is smooth. For Takens' theorem, h must be "generic" (a residual set in C^2). The question is whether cosine-of-attention-divergence satisfies this or whether it has degenerate level sets that break injectivity of the delay map.

**Why it matters:** Without this argument, the connection between SAD trajectories and attractor reconstruction is heuristic rather than principled. The honest answer may be: "genericity cannot be proved without knowing the attractor's geometry, but it can be tested empirically via embedding continuity and false-nearest-neighbor statistics." That is still worth writing down explicitly.

### Priority 4: Flower graph / belief state correspondence

*Mathematical work.*

The proof connecting fd-formalization's (u,v)-flower graph construction to non-unifilar HMM branching structure. This closes the formal loop from existence theory through fractal geometry to empirical observation.

**Why it matters:** If this proof works, the full chain is: formal dimension theory (Lean 4, fd-formalization) --- predicted fractal dimension --- empirical measurement (navi-SAD PE + navi-fractal box-counting) --- comparison. A formally verified prediction tested by an empirically validated instrument. See [Three-Repo Unification](three-repo-unification.md).

## Near-term: Instrument hardening

### Priority 5: Polish pass

PerStepDict boundary type, fail-closed fixes, type annotations, named constants, CI coverage for scripts/, test gaps, CLAUDE.md refresh. Spec and plan complete (`docs/plans/POLISH_PASS_SPEC.md`, `docs/plans/POLISH_PASS_PLAN.md`). This is housekeeping that makes everything downstream more reliable.

### Priority 6: Renyi fingerprint

Port Renyi entropy parameter sweeps (q in [0.1, 7.0]) and Renyi complexity (Jensen-Renyi divergence from uniform) from the production C++ kernel (`navi_dsc_renyi.h`). Shannon PE becomes the q=1 special case. The full (H_q, C_q) fingerprint curve per head characterizes the *shape* of the ordinal pattern distribution --- the attractor's ordinal signature.

**Why it matters:** Two heads with identical Shannon PE can have completely different Renyi fingerprint shapes. Low q emphasizes rare patterns (attractor diversity); high q emphasizes dominant patterns (collapse concentration). Attractor collapse is not just "entropy drops" --- the fingerprint tells you *how* it collapses.

### Priority 7: Permutation null test

Stratified permutation null on the recurrence statistic with eligibility accounting. Analysis module under `src/navi_sad/analysis/`. Spec and plan complete (`docs/plans/PE_RECURRENCE_NULL_PLAN.md`).

**Why it matters:** The 338/1024 heads and 4.6:1 asymmetry from the pilot are meaningless without a null distribution. The permutation test determines whether the observed pattern is distinguishable from what you would get by shuffling labels while preserving class sizes. If it is not, the pilot signal is a multiple-comparison artifact.

## Gate 3: Synthetic benchmark

### Priority 8: Synthetic HMM benchmark harness

*Planned.* Build a family of binary HMMs with known unifilarity properties, ranging from fully unifilar (point attractor) to maximally non-unifilar (known fractal dimension from process structure). Generate matched-length sequences, feed to Mistral with next-token prediction framing, capture per-head SAD trajectories.

**Why it matters:** This is the experimental setup for Gate 3. Synthetic processes give ground truth --- you know the fractal dimension of the belief state because you designed the generating process.

### Priority 9: Gate 3 --- rank correlation

*Planned.* Per-head PE (across D=3..7) correlated with known fractal dimension of the generating process. Pass criterion: significant Spearman rank correlation in L15--21 heads, surviving permutation null.

**Why it matters:** This is the central prediction. If per-head PE tracks the computational-mechanical complexity of the inference problem --- as measured by the fractal dimension of the belief state geometry --- then the instrument is measuring what the theory says it should measure. If it does not, the theoretical framing is wrong or the instrument is insensitive.

## Post-validation: Natural language

### Priority 10: Natural language benchmarks

*Planned. Only after Gate 3 passes.*

Apply the validated instrument to controlled natural-language tasks: nested syntactic dependencies, multi-step reasoning with hidden variables, ambiguous pronoun resolution. These are designed to vary the complexity of the inference problem while staying within natural language, bridging from synthetic HMMs to realistic use cases.

### Priority 11: TruthfulQA revisited

*Planned. Only after Gate 3 passes.*

Reframe the original pilot data with the validated hypothesis. Not correctness separation, but belief-state complexity characterization. The instrument measures attractor geometry; TruthfulQA provides one regime partition among many.

## Theoretical integration

### Priority 12: Three-repo unification

*Theoretical work. Ongoing.*

cd-formalization (existence theory), fd-formalization (fractal dimension theory), navi-SAD (empirical reconstruction). One instrument suite on one state space. The Shai et al. result provides the empirical anchor: transformers construct belief state geometry in their residual streams, and that geometry can be genuinely fractal.

**Why it matters:** This is the long-term vision. Three mathematical frameworks --- autopoietic coherence conditions, box-counting fractal dimension formalized in Lean 4, and empirical attractor reconstruction via delay-coordinate embedding --- converging on the same state space. See [Three-Repo Unification](three-repo-unification.md) for the full argument.
