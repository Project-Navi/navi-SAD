# Three-Repo Unification

*Status: Theoretically motivated. In progress.*

*Last updated: 2026-03-27*

Three Project Navi repositories --- [cd-formalization](https://github.com/Project-Navi/cd-formalization), [fd-formalization](https://github.com/Project-Navi/fd-formalization), and [navi-SAD](https://github.com/Project-Navi/navi-SAD) --- address the same underlying question at different levels of abstraction: what is the geometry of a system's internal state when it processes information, and can we characterize that geometry formally and measure it empirically?

This page describes the theoretical chain connecting them and is honest about which links are proved, which are predicted, and which remain open.

---

## The three layers

### Layer 1: Existence theory (cd-formalization)

[cd-formalization](https://github.com/Project-Navi/cd-formalization) is a Lean 4 formalization of the Creative Determinant framework. It proves that self-sustaining coherent configurations --- autopoietic closure --- can exist under certain conditions on a compact Riemannian manifold.

The core result: when the viability potential $b(x) = \kappa\gamma - \lambda\mu$ (care-coherence support minus contradiction cost) exceeds a spectral threshold, nontrivial solutions to the nonlinear elliptic boundary value problem exist (Theorem 3.16). Fifteen theorems, zero `sorry`, five explicit axioms packaging classical PDE results not yet in Mathlib (`PDEInfra`).

**What this layer says:** Under sufficient conditions, systems that sustain their own coherence through internal dynamics *can* exist. This is an existence claim, not a measurement claim.

**Verification status:** Machine-checked. `lake build --wfail` is the arbiter.

### Layer 2: Fractal dimension theory (fd-formalization)

[fd-formalization](https://github.com/Project-Navi/fd-formalization) formalizes the fractal dimension of (u,v)-flower networks in Lean 4. It proves that the log-ratio of vertex count to hub distance converges:

$$
\lim_{g \to \infty} \frac{\log |V_g|}{\log L_g} = \frac{\log(u + v)}{\log u} \quad \text{for } 1 < u \le v
$$

The proof establishes: vertex count recurrence, hub distance $L_g = u^g$, two-sided squeeze bounds, and the limit via the squeeze theorem. The F2 bridge additionally proves that `SimpleGraph.dist hub0 hub1 = u^g` on the explicit graph construction, connecting the arithmetic formula to graph-theoretic distance.

**What this layer says:** For a family of self-similar recursive graphs, the fractal dimension is analytically known and formally verified. Zero `sorry`, zero custom axioms.

**Verification status:** Machine-checked. Same standard as Layer 1.

[navi-fractal](https://github.com/Project-Navi/navi-fractal) is the empirical companion: it measures sandbox (mass-radius) dimension on finite graphs and calibrates against the analytical dimensions that fd-formalization proves. The gap between sandbox estimates and analytical ground truth is documented openly, not hidden (see navi-fractal's [calibration regime](https://project-navi.github.io/navi-fractal/explanation/calibration-regime/) and [theory bridge](https://project-navi.github.io/navi-fractal/explanation/theory-bridge/)).

### Layer 3: Empirical reconstruction (navi-SAD)

[navi-SAD](https://github.com/Project-Navi/navi-SAD) operationalizes the measurement. It runs softmax and linear attention in parallel on the same frozen weights, computes per-head [cosine divergence](../theory/sad-instrument.md) at each generation step, and treats the resulting scalar trajectories as [delay-coordinate embeddings](../theory/takens-embedding.md) of the model's internal dynamical state. [Permutation entropy](../reference/glossary.md#permutation-entropy-pe) on these trajectories characterizes the attractor's complexity.

**What this layer says:** Given a running transformer, here is what we observe about per-head attractor structure (in the sense of Sauer, Yorke & Casdagli, 1991: a compact invariant set characterized by its box-counting dimension) during inference.

**Verification status:** Instrument validation (Gates 0--2) [complete](../instrument/gate-discipline.md). Attractor characterization is theoretically motivated but not yet empirically grounded. Gate 3 tests the critical prediction.

---

## The bridge: Shai et al. ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943))

The theoretical anchor connecting these layers is Shai et al., "Transformers Represent Belief State Geometry in their Residual Stream" ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)). Their key results:

1. **Transformers construct belief state geometry in their residual streams.** The belief state --- the posterior distribution over hidden states given the observed sequence --- is linearly recoverable from transformer hidden states. The geometry is real, not hypothetical.

2. **That geometry can be genuinely fractal.** For non-unifilar inference processes (where the hidden state cannot be uniquely determined from the observations), the belief state space has fractal structure. Shai et al. demonstrated Sierpinski-triangle-like attractors for specific HMM families.

3. **The fractal dimension depends on the generating process.** Unifilar processes produce point attractors (trivial geometry). Non-unifilar processes produce attractors whose fractal dimension is determined by the process structure.

4. **Geometry is distributed across layers.** Shai et al. find that belief state geometry is "represented in the final residual stream or distributed across the residual streams of multiple layers." The mechanism of this distribution is analyzed in the companion work (Piotrowski et al., NeurIPS 2024). *[Whether this produces progressive depth-increasing separation in per-head PE is testable with existing pilot data. Not yet tested.]*

---

## The argument chain

The following chain connects the three layers through the Shai et al. bridge. Each link is annotated with its epistemic status.

### Link 1: Existence $\to$ state space

The CD framework characterizes conditions under which coherent configurations exist on a compact Riemannian manifold. The belief state space of a transformer (as characterized by Shai et al.) is one candidate instantiation of this state space.

*Status: Analogical. The CD framework is a general existence theory. Its application to transformer belief states is a proposed interpretation, not a proved correspondence. The belief state space of a non-unifilar process is a compact set but not generally a smooth manifold --- it can be fractal. The relationship between the CD framework's smooth manifold setting and fractal attractor geometry is an open problem.*

### Link 2: Fractal geometry $\to$ belief state structure

fd-formalization proves the fractal dimension of (u,v)-flower networks. Shai et al. demonstrate that non-unifilar HMMs produce fractal belief state geometry. If the branching structure of a non-unifilar HMM's belief state space maps to a specific (u,v)-flower construction, the analytically known dimension gives a predicted dimension for the belief state attractor.

*Status: Predicted but unproved. The connection between HMM branching structure and flower graph topology is mathematical work that has not been completed. This is Roadmap item 4 ("Flower graph / belief state correspondence"). If this proof works, it closes the formal loop from fractal dimension theory to empirical prediction. If it doesn't, the connection between fd-formalization and empirical observation remains indirect (calibration only, not predictive).*

### Link 3: Observable $\to$ attractor reconstruction

SAD's per-head cosine divergence is a smooth function of the residual stream state. The argument:

- Residual stream state $x_t$ encodes belief state (Shai et al., empirical)
- Attention computation is a deterministic function of $x_t$
- Per-head softmax and linear attention outputs are deterministic functions of $x_t$
- Cosine divergence is smooth
- Therefore SAD delta $\delta(t) = h(x_t)$ where $h$ is a composition of smooth functions

For Takens-style reconstruction, $h$ must be "generic" (a residual set in $C^2$). Whether cosine-of-attention-divergence satisfies this, or whether it has degenerate level sets that break injectivity of the delay map, cannot be proved without knowing the attractor's geometry. It can be tested empirically via embedding continuity and false-nearest-neighbor statistics.

*Status: Theoretically motivated. The smoothness argument is sound. Genericity cannot be proved a priori --- it is an empirical question. The honest statement: SAD is a smooth observable of the residual stream state. Whether it is a generic observable is testable but not yet tested.*

Note: Shai et al. recovered belief state geometry from the residual stream via linear probing. SAD's observable is cosine divergence between attention paths, not the residual stream directly. Attention computation is a deterministic function of the residual stream, so divergence inherits continuity. But "smooth" and "generic" are different properties. The relationship is plausible but must be stated explicitly, not assumed.

### Link 4: PE $\to$ attractor complexity

Permutation entropy on delay-coordinate embeddings measures the complexity of the reconstructed attractor's ordinal pattern distribution. If the embedding is sufficient (enough dimensions to unfold the attractor), PE tracks the dynamical complexity of the underlying system.

*Status: Theoretically grounded. Bandt-Pompe ordinal patterns are established tools for delay-coordinate reconstructions (Bandt & Pompe, Physical Review Letters, 2002). The open question is embedding dimension sufficiency: if the attractor's fractal dimension exceeds what D=3 can reconstruct, PE shows artificial inflation. Testable via D-sweep (Roadmap item 1). Not yet tested on pilot data.*

### Link 5: PE $\to$ fractal dimension (Gate 3)

The critical prediction: per-head PE should correlate with the known fractal dimension of the generating process. Synthetic HMMs with analytically known unifilarity properties and fractal dimensions provide the ground truth.

*Status: Untested. This is [Gate 3](../instrument/gate-discipline.md). Pass criterion: significant Spearman rank correlation between per-head PE and known fractal dimension in L15--21 heads, surviving permutation null. This replaces the original TruthfulQA AUROC criterion.*

---

## What validation looks like

If the full chain validates:

1. **Existence theory** says coherent self-sustaining configurations *can* exist under spectral conditions (cd-formalization, proved)
2. **Fractal dimension theory** characterizes the geometry of self-similar recursive structures with machine-verified formulas (fd-formalization, proved)
3. **Empirical calibration** confirms that sandbox dimension estimates converge toward the analytical ground truth (navi-fractal, demonstrated with documented gaps)
4. **Instrument validation** confirms that the measurement apparatus does not disturb the system and produces stable, reproducible outputs (navi-SAD Gates 0--2, passed)
5. **Signal validation** confirms that per-head PE tracks the fractal dimension of the inference problem's belief state (Gate 3, *untested*)

That would be a formal-to-empirical pipeline: existence $\to$ geometry $\to$ observation. Formally verified predictions tested by an empirically validated instrument.

If Gate 3 fails, the pipeline is broken at the empirical end. The formal results stand independently (Lean doesn't care about experimental outcomes), but the claim that SAD observes belief state geometry collapses. The instrument would still be a valid measurement of per-head attention dynamics --- it just wouldn't be measuring what the theory predicts.

---

## Open gaps

These are the unsolved problems in the chain, roughly ordered by how much they block progress.

**Flower graph / belief state correspondence.** The missing proof connecting fd-formalization's (u,v)-flower construction to non-unifilar HMM branching structure. Without this, the link between formal fractal dimension and belief state geometry is indirect. (Roadmap item 4)

**Observable genericity.** The explicit justification that per-head SAD divergence is a generic observable of the belief state in the Takens sense. The smoothness argument is straightforward; genericity requires either a mathematical proof (unlikely without full attractor characterization) or empirical testing via false-nearest-neighbor statistics. (Roadmap item 3)

**Embedding dimension sufficiency.** Whether D=3 (or D=4, D=5) is enough to unfold the attractor. Testable now via D-sweep on existing pilot data. (Roadmap item 1)

**Progressive construction.** Whether per-layer PE separation grows from early to late layers, as predicted by Shai et al.'s progressive construction finding. Testable now on pilot data. (Roadmap item 2)

**Mane generalization.** The belief state attractor for non-unifilar processes is a compact set, not a smooth manifold. Standard Takens' theorem applies to smooth manifolds. The Mane embedding theorem generalizes to compact sets, but the precise conditions and their applicability to SAD's observation function have not been worked through.

**CD instantiation.** Whether the CD framework's semiotic manifold formulation applies to transformer belief states. This is the most speculative link in the chain --- it requires showing that the PDE setting (smooth manifold, elliptic operator) connects meaningfully to the fractal attractor setting. This may require extending the CD framework or accepting that the connection is analogical rather than formal.

---

## The honest version

The three repositories share intellectual parentage and a common theoretical motivation, but they do not yet form a single validated instrument suite. The formal layers (cd-formalization, fd-formalization) are independently verified. The empirical layer (navi-SAD) is validated as an instrument but not yet as a probe of belief state geometry. The links between them range from machine-checked (fractal dimension formulas calibrated against empirical estimates) to speculative (CD framework instantiation on transformer state spaces).

The reason to pursue this is that the pieces exist in a configuration where falsification is possible. Gate 3 is designed to test the most load-bearing prediction: that PE on per-head SAD trajectories tracks the fractal dimension of the generating process. If it does, the chain from formal dimension theory through empirical measurement holds. If it doesn't, we know which link broke and can investigate why.

Three repos. One state space. The distance between them is now smaller than the distance within any one of them --- but distance is not zero, and some of the bridges are still under construction.

---

## Repository links

| Repository | Layer | Verification | Status |
|-----------|-------|-------------|--------|
| [cd-formalization](https://github.com/Project-Navi/cd-formalization) | Existence theory | Lean 4, zero sorry, 5 explicit axioms | Complete |
| [fd-formalization](https://github.com/Project-Navi/fd-formalization) | Fractal dimension | Lean 4, zero sorry, zero custom axioms | Complete |
| [navi-fractal](https://github.com/Project-Navi/navi-fractal) | Empirical dimension estimation | Calibrated against fd-formalization | Complete |
| [navi-creative-determinant](https://github.com/Project-Navi/navi-creative-determinant) | CD framework (paper + numerics + formalization) | Paper + Lean 4 + notebooks | Complete |
| [navi-SAD](https://github.com/Project-Navi/navi-SAD) | Empirical attractor reconstruction | Gates 0--2 passed; Gate 3 pending | In progress |

## Key references

- Shai, A. S., Marzen, S. E., Teixeira, L., Gietelink Oldenziel, A., & Riechers, P. M. (2024). Transformers Represent Belief State Geometry in their Residual Stream. *NeurIPS 2024*. [arXiv:2405.15943](https://arxiv.org/abs/2405.15943). [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html).
- Rozenfeld, H. D., Havlin, S. & ben-Avraham, D. (2007). Fractal and transfractal recursive scale-free nets. *New Journal of Physics*, 9:175.
- Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. *Physical Review Letters*, 88(17), 174102.
- Han, D., et al. (2024). Bridging the Divide: Reconsidering Softmax and Linear Attention. [arXiv:2412.06590](https://arxiv.org/abs/2412.06590).
- Spence, N. (2026). The Creative Determinant: Autopoietic Closure as a Nonlinear Elliptic Boundary Value Problem with Lean 4-Verified Existence Conditions. *Project Navi LLC*.
