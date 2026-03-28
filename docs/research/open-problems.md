# Open Problems

*Last updated: 2026-03-27*

*Status: Active. Contributions welcome.*

---

This document lists open problems in navi-SAD, tiered by difficulty and resource requirements. Each problem has ground truth on why it matters, what you would need, and what success looks like. Negative results count --- showing that a proposed approach does not work is valuable information.

The tiers follow the [Neel Nanda pattern](https://www.neelnanda.io/mechanistic-interpretability/getting-started): accessible problems that anyone can pick up, intermediate problems requiring more tooling, and research-grade problems that are open questions.

---

## Accessible Tier

*No GPU required. Existing pilot data. Can start today.*

### 1. D-sweep on pilot data

**Status:** Open. *Theoretically motivated.*

**Problem:** The pilot computed permutation entropy at embedding dimension D=3 only. Does the signal change at D=4? Under the 2*D! eligibility policy, D=3 requires 12-step minimum windows and D=4 requires 48-step windows. Pilot trajectories are ~77 steps, so D=4 is feasible. D=5 requires 240-step windows --- infeasible on pilot data.

**Why it matters:** If PE stabilizes between D=3 and D=4, the delay-coordinate embedding has enough dimensions to capture the attractor's topology and the pilot findings hold at the existing embedding dimension. If PE changes substantially, D=3 is undersampling the attractor and the 338-head signal may be an artifact of insufficient embedding dimension. This directly tests whether the [Takens' reconstruction](../theory/takens-embedding.md) is well-posed for these trajectories.

**What you need:** The pilot artifacts (`results/pilot_gate3/samples.json` --- gitignored, available on request), Python, `navi_sad.signal.pe_features`. Call `compute_sample_pe_features` with `PEConfig(D=d)` for d in {3, 4}. Compare PE distributions per head across D values.

**What success looks like:** A clear report on PE stability vs D: either "PE stabilizes by D=4, confirming embedding sufficiency" or "PE shifts substantially at D=4, indicating D=3 is insufficient and longer sequences are needed for Gate 3."

---

### 2. Layer-stratified PE profiles

**Status:** Open. *Theoretically motivated.*

**Problem:** The pilot found per-head PE signal, but did not analyze whether it concentrates in specific layers. Shai et al. (NeurIPS 2024) predicts that later layers have more fully constructed belief state geometry --- so correct/incorrect PE separation should grow with depth from L0 to L31.

**Why it matters:** If separation grows from early to late layers, it is convergent evidence that [SAD](../theory/sad-instrument.md) observes belief state construction progressing through the layer stack. If separation concentrates in L15--21, it matches the pilot's recurring-head zone and the progressive construction prediction. If separation is flat across layers, progressive construction does not apply to this observable, which would weaken the theoretical motivation.

**What you need:** Same pilot artifacts as problem 1. Group PE results by layer, compute Cohen's d (correct vs incorrect, using the guarded implementation in `navi_sad.pilot.helpers`) per layer, plot the 32-point profile.

**What success looks like:** A layer profile showing either depth-increasing separation (supports progressive construction), mid-late concentration (supports L15--21 prediction), or flat/random structure (falsifies this specific prediction).

---

## Intermediate Tier

*May require code work. May require GPU for validation runs. Builds on existing infrastructure.*

### 3. Observable genericity argument

**Status:** Open. *Theoretically motivated.*

**Problem:** For [Takens' embedding theorem](../theory/takens-embedding.md) to apply, the observable function h must be "generic" --- a residual set in C^2. SAD delta is the cosine divergence between softmax and linear attention outputs, both deterministic functions of the residual stream state. Is this composition generic, or does it have degenerate level sets that break injectivity of the delay map?

**Why it matters:** This is the theoretical gap between "SAD trajectories look like they reconstruct an attractor" and "Takens' theorem guarantees that they do." Without this argument, the connection between per-head SAD and attractor reconstruction is heuristic. The honest outcome may be: "genericity cannot be proved without knowing the attractor's geometry, but it can be tested empirically" --- and that is still worth stating explicitly.

**What you need:** Knowledge of smooth dynamical systems and Takens' embedding theory. The argument chain to formalize: residual stream encodes belief state (Shai et al.) --- attention is a deterministic function of it --- cosine divergence is smooth --- SAD delta is h(x_t) where h is smooth --- but is h generic? Empirical testing via embedding continuity and false-nearest-neighbor statistics on pilot data would supplement the theoretical argument.

**What success looks like:** Either a proof that cosine-of-attention-divergence is generic for the relevant function class, a precise statement of the additional assumption required, or an empirical protocol that tests genericity on real data.

---

### 4. Renyi fingerprint port

**Status:** Open. *Planned.*

**Problem:** Shannon PE is a single number characterizing the ordinal pattern distribution. The Renyi entropy family (parametrized by q in [0.1, 7.0]) characterizes the full *shape* of that distribution. The production C++ kernel (`navi_dsc_renyi.h`) computes Renyi entropy, Renyi complexity (Jensen-Renyi divergence from uniform), and parametric fingerprint curves. This needs to be ported to the Python signal layer.

**Why it matters:** Two heads with identical Shannon PE can have completely different Renyi fingerprint shapes. Low q emphasizes rare ordinal patterns (attractor diversity); high q emphasizes dominant patterns (collapse concentration). The \( (H_q, C_q) \) fingerprint curve per head is the attractor's ordinal signature --- it tells you not just *how much* complexity there is, but *what kind*. The complexity-entropy plane was introduced by Rosso et al. (*Physical Review Letters* 99, 154102, 2007) as a joint diagnostic for distinguishing noise from chaos from periodic dynamics using Bandt-Pompe patterns. Attractor collapse in confabulation may show up as fingerprint shape change even when Shannon PE is unchanged.

**What you need:** Python, the existing `signal/ordinal.py` infrastructure (pattern counts are already computed), and the reference C++ kernel for validation. Implementation path: add `renyi_entropy(pattern_counts, D, q)`, `renyi_complexity(pattern_counts, D, q)`, and `renyi_fingerprint(pattern_counts, D, q_range)` to `signal/ordinal.py`, all operating on the same pattern distribution that `permutation_entropy` already produces. Extend `pe_features.py` to compute fingerprint curves per head.

**What success looks like:** A validated Python port that reproduces the C++ kernel's output on reference inputs, with per-head fingerprint curves computable on pilot data. Comparison of fingerprint shape between correct and incorrect groups, supplementing the scalar PE analysis.

---

### 5. Permutation null test

**Status:** Open. *Planned.*

**Problem:** The pilot found 338/1024 heads with |d|>0.5 and a 4.6:1 directional asymmetry. These numbers are meaningless without a null distribution. The permutation test shuffles correct/incorrect labels while preserving class sizes, recomputes per-head statistics, and determines whether the observed pattern is distinguishable from chance.

**Why it matters:** With 1024 heads times multiple modes and segments, multiple-comparison inflation is a real concern. The permutation null directly answers: "could this signal arise from random label assignment?" If the observed recurrence count and asymmetry ratio fall within the null distribution, the pilot signal is an artifact. If they fall outside, the signal is robust to relabeling.

**What you need:** Pilot artifacts, the analysis module (to be built under `src/navi_sad/analysis/`), and a clear specification of the test statistic. The experimental design is specified in the repository (see the internal plans directory). The implementation requires eligibility accounting --- heads that are ineligible for PE at a given D must be excluded from the null as well as the observed statistic.

**What success looks like:** A permutation p-value for the recurrence count (338/1024) and the asymmetry ratio (4.6:1). Either "p < 0.05, the signal survives relabeling" or "p > 0.05, the signal is consistent with multiple-comparison noise." Both outcomes are useful.

---

## Research-Grade Tier

*Open questions. May require significant GPU time, new code infrastructure, or novel experimental design.*

### 6. Gate 3 --- synthetic HMM benchmark

**Status:** Open. *Planned.*

**Problem:** Build a family of binary HMMs with known unifilarity properties, ranging from fully unifilar (point attractor, zero fractal dimension) to maximally non-unifilar (known fractal dimension computable from process structure). Generate matched-length sequences, feed to Mistral with next-token prediction framing, capture per-head SAD trajectories, and test for rank correlation between per-head PE and known fractal dimension.

**Why it matters:** This is the central validation of the [Takens' embedding framing](../theory/takens-embedding.md). If per-head PE tracks the computational-mechanical complexity of the generating process --- as measured by the fractal dimension of its belief state geometry --- then the instrument is measuring what the theory predicts. If it does not, either the theory is wrong, the instrument is insensitive, or the observable is not generic. Unlike TruthfulQA, synthetic HMMs provide exact ground truth on the quantity being measured.

**What you need:** GPU access (Mistral-7B inference), HMM design with known fractal dimension calculations (background in computational mechanics helps), the full navi-SAD instrument stack (Gates 0--2 passing), and the permutation null infrastructure from problem 5. Experimental design: a family of 5--10 HMMs spanning the unifilarity spectrum, matched-length sequences (~256 tokens), per-head PE at D=3..7.

**What success looks like:** Significant Spearman rank correlation between per-head PE and known fractal dimension in L15--21 heads, surviving permutation null. This is the Gate 3 pass criterion. Failure would mean that per-head PE does not track process complexity, which would force a fundamental reassessment of the theoretical framing.

---

### 7. Position-aware SAD normalization

**Status:** Open. Deferred (IA audit finding F-09).

**Problem:** Linear attention's denominator grows with prefix length, causing SAD deltas to increase mechanically with generation position. Both groups (correct and incorrect) show deltas climbing from ~0.24 to ~0.40 over generation. First-differencing removes the trend empirically, but an analytical normalization --- dividing by the expected divergence growth given the prefix length --- would be more principled.

**Why it matters:** First-differencing is a pragmatic fix. It removes the position trend but also destroys some temporal structure. An analytical normalization would preserve the full trajectory shape while correcting for the known position confound. This matters most for long generations where the denominator growth dominates the signal.

**What you need:** Analysis of the linear attention denominator's growth rate as a function of prefix length and value distribution. May require deriving the expected cosine divergence under null assumptions (random softmax vs linear attention weights). Validation that the normalization does not introduce new artifacts.

**What success looks like:** A normalization function that removes the position trend from SAD deltas without destroying per-head temporal structure. Validated by showing that normalized deltas are stationary (no position trend) while preserving the per-head PE signal observed in the pilot.

---

## Theoretical Tier

*Mathematical work. May require proof assistance, domain expertise in computational mechanics or formal methods.*

### 8. Flower graph / belief state correspondence

**Status:** Open. *Theoretically motivated.*

**Problem:** The fd-formalization repository formalizes box-counting fractal dimension for (u,v)-flower graphs in Lean 4. Shai et al. showed that non-unifilar HMMs produce fractal belief state geometry. The bridge: show that the branching structure of a non-unifilar HMM's belief state space maps to a specific (u,v)-flower construction, so the analytically known dimension from fd-formalization gives the predicted dimension for the belief state attractor.

**Why it matters:** If this proof works, the full chain is: formal dimension theory (Lean 4) --- predicted fractal dimension --- empirical measurement (navi-SAD PE + navi-fractal box-counting) --- comparison. A formally verified prediction tested by an empirically validated instrument. This is the mathematical result that nobody in mechanistic interpretability currently has the tools to produce.

**What you need:** Background in computational mechanics (epsilon-machines, mixed-state presentations), fractal geometry (box-counting dimension, self-similar constructions), and Lean 4 formalization. The fd-formalization repo contains the flower graph dimension proofs. The Shai et al. paper provides the empirical anchor. The gap is the mapping between HMM branching structure and flower graph parameters (u, v).

**What success looks like:** A proof (or a precise conjecture with supporting evidence) that for a specific class of non-unifilar HMMs, the belief state attractor has box-counting dimension equal to the (u,v)-flower dimension for computable (u,v). Even a partial result --- for a single HMM family --- would be significant.

---

### 9. Three-repo unification proof

**Status:** Open. *Theoretically motivated.*

**Problem:** Three repositories --- cd-formalization (existence theory), fd-formalization (fractal dimension theory), and navi-SAD (empirical reconstruction) --- are three perspectives on one state space. The unification requires showing that cd-formalization's coherence conditions, fd-formalization's dimension theory, and navi-SAD's delay-coordinate embedding all operate on the same mathematical object: the belief state geometry that Shai et al. showed transformers construct in their residual streams.

**Why it matters:** This is the long-term vision of the project. Each repo solves a piece: cd-formalization asks "can this system sustain autopoietic closure?" fd-formalization asks "what does the fractal geometry look like?" navi-SAD asks "what is the system actually doing during inference?" Unifying them would connect existence theory, measurement theory, and empirical observation into a single coherent framework for understanding inference-time computation. See [Three-Repo Unification](three-repo-unification.md).

**What you need:** Familiarity with all three repos, background in computational mechanics, PDE theory (for cd-formalization's elliptic equations), and formal methods (Lean 4, for fd-formalization). The Shai et al. result is the empirical anchor. The Mane embedding theorem (which generalizes Takens to compact sets, not just manifolds) provides the mathematical bridge for fractal attractors.

**What success looks like:** A precise mathematical statement of how the three frameworks relate, with either a proof of consistency or a clear identification of the assumptions that must hold for consistency. Even identifying the exact technical conditions under which the unification fails would be a valuable contribution.

---

## How to contribute

If you make progress on any of these problems, please:

1. Open an issue tagged `open-problem` referencing the problem number.
2. Share your results --- even partial, even negative.
3. If you resolve a problem, open a PR adding your result or a linked document.

**Negative results count.** If you show that D=3 is insufficient, or that the permutation null kills the pilot signal, or that the genericity argument requires an assumption that does not hold --- that is valuable information that redirects the research.

## Proposing new open problems

If you identify a gap not listed here, open an issue tagged `open-problem-proposal` with:

- A clear statement of the problem.
- Why it matters.
- Suggested starting point (if any).

We will add it if it fits the research framework.
