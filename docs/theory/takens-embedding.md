# Takens' Embedding & Attractor Reconstruction

*Status: Theoretically motivated. Empirical validation planned (Gate 3).*

Each per-head [SAD trajectory](sad-instrument.md) is treated as a delay-coordinate embedding of the model's internal dynamical state. We are not measuring a signal --- we are reconstructing an attractor (a compact invariant set in the sense of Sauer, Yorke & Casdagli, 1991). [Permutation entropy](../reference/glossary.md#permutation-entropy-pe) is load-bearing: Bandt-Pompe ordinal patterns are designed for delay-coordinate reconstructions.

---

## The intuition: reconstructing what you cannot see

Suppose a system has internal state you cannot directly observe --- you can only measure some scalar quantity at each time step. Delay-coordinate embedding is the claim that if you collect enough consecutive measurements and arrange them as vectors, the geometry of those vectors faithfully reconstructs the geometry of the hidden state.

Concretely: take a scalar time series \( x_1, x_2, x_3, \ldots \) and form vectors by stacking consecutive values:

\[
\mathbf{v}_t = (x_t,\; x_{t+\tau},\; x_{t+2\tau},\; \ldots,\; x_{t+(D-1)\tau})
\]

where \( D \) is the embedding dimension and \( \tau \) is the delay. Each vector \( \mathbf{v}_t \) is a point in \( \mathbb{R}^D \). Plot all of them together and you get a *reconstructed attractor* --- a geometric object that preserves the topological structure of the original system's dynamics, even though you never had direct access to the full state.

This is not an approximation or a heuristic. Takens' theorem (1981) proves that for generic observation functions, the reconstruction is a *diffeomorphism* --- a smooth, invertible map --- onto the original attractor. The reconstructed geometry preserves periodic orbits, Lyapunov exponents, entropy, and other dynamical invariants. It does not preserve the shape (two reconstructions from different observables will look different), but the topological and dynamical content is identical.

For SAD, the scalar time series is the per-head cosine divergence between softmax and linear attention at each generation step. The delay vectors are constructed from consecutive divergence values. The resulting attractor reconstruction captures the temporal dynamics of that head's attention behavior during generation --- how its divergence evolves, not just its mean.

## Formal statement

**Takens' Embedding Theorem** (Takens, 1981). Let \( M \) be a compact smooth manifold of dimension \( m \), let \( \varphi: M \to M \) be a \( C^2 \) diffeomorphism, and let \( h: M \to \mathbb{R} \) be a \( C^2 \) observation function. Define the delay-coordinate map:

\[
\Phi_{(\varphi, h)}(x) = \bigl(h(x),\; h(\varphi(x)),\; h(\varphi^2(x)),\; \ldots,\; h(\varphi^{2m}(x))\bigr)
\]

For a *generic* pair \( (\varphi, h) \) --- that is, for an open and dense subset of all such pairs --- the map \( \Phi_{(\varphi, h)}: M \to \mathbb{R}^{2m+1} \) is an embedding (injective immersion with a smooth inverse onto its image).

In practical terms: if the attractor lives on an \( m \)-dimensional manifold, an embedding dimension of \( 2m + 1 \) suffices to reconstruct it faithfully from a single scalar observable, provided the observable is "generic" --- meaning it does not have special symmetries that make it blind to parts of the dynamics.

**Extensions.** Sauer, Yorke, and Casdagli (1991) generalized the theorem to attractors with arbitrary box-counting dimension \( d_A \), requiring only \( D > 2d_A \) rather than \( 2m + 1 \). This is the version most relevant to fractal attractors, where the attractor dimension is typically much smaller than the ambient manifold dimension.

## Why ordinal patterns are the right measurement

Given a delay-coordinate reconstruction, you need a way to quantify its complexity. You could compute correlation dimension, Lyapunov exponents, or other geometric invariants --- but these require long time series (thousands of points) and careful estimation. SAD trajectories during generation are short: tens to hundreds of steps. We need a measure that works on short, noisy sequences.

Bandt and Pompe (2002) introduced *ordinal patterns* for exactly this setting. Instead of measuring the *magnitudes* of values in each delay vector, you record only their *rank order*. For an embedding dimension \( D = 3 \), the delay vector \( (x_t, x_{t+1}, x_{t+2}) \) is classified by which of the 3! = 6 possible orderings its values take. For example, if \( x_t < x_{t+2} < x_{t+1} \), the ordinal pattern is \( (0, 2, 1) \).

This is a natural measurement for delay-coordinate reconstructions for three reasons:

1. **Invariance to monotonic transforms.** Ordinal patterns depend only on the *ordering* of values, not their scale or distribution. This makes them robust to the nonlinear distortions that different observation functions introduce --- exactly the kind of distortion Takens' theorem says is harmless.

2. **Robustness to noise.** Small perturbations rarely change rank orders. A noisy measurement that shifts a value by 0.001 will almost never swap the ordering of two values that differ by 0.1. This is critical for short instrument trajectories where noise is a large fraction of the signal.

3. **Computability on short sequences.** Ordinal patterns can be extracted from any sequence of length \( n \geq D \). The number of distinct patterns is \( D! \), which is small enough for reliable frequency estimation even with tens of windows. Compare this to correlation dimension estimation, which typically needs \( 10^3 \)--\( 10^4 \) points.

**Tie handling.** When two values in a delay vector are equal (or within machine epsilon), the rank order is ambiguous. Our implementation *excludes* tied windows rather than imposing an arbitrary tiebreaker. This prevents fake ordinal structure from inflating entropy where the dynamics are genuinely degenerate. See the [module reference](../reference/module-reference.md) for implementation details.

## Permutation entropy

Permutation entropy (PE) is the Shannon entropy of the ordinal pattern frequency distribution. Given a time series and embedding parameters \( (D, \tau) \), count the frequency of each ordinal pattern \( \pi_i \) across all windows, normalize to get probabilities \( p(\pi_i) \), and compute:

\[
H(D) = -\sum_{i=1}^{D!} p(\pi_i) \ln p(\pi_i)
\]

The normalized form divides by the maximum possible entropy:

\[
PE(D) = \frac{H(D)}{\ln(D!)}
\]

| Term | Meaning |
|------|---------|
| \( D \) | Embedding dimension --- the length of each ordinal pattern. Determines the resolution: \( D! \) possible patterns. |
| \( \tau \) | Embedding delay --- the step size between consecutive elements in each delay vector. \( \tau = 1 \) means adjacent steps. |
| \( p(\pi_i) \) | Relative frequency of the \( i \)-th ordinal pattern across all windows in the time series. |
| \( H(D) \) | Raw permutation entropy (in nats, when using natural log). |
| \( \ln(D!) \) | Maximum possible entropy --- achieved when all \( D! \) patterns are equally likely (i.e., the series is indistinguishable from white noise at this resolution). |
| \( PE(D) \) | Normalized permutation entropy, in \( [0, 1] \). |

**Interpretation.** \( PE = 1 \) means all ordinal patterns appear equally often --- the trajectory is maximally complex at this embedding dimension, indistinguishable from randomness. \( PE = 0 \) means only a single pattern ever appears --- the trajectory is completely predictable. Values in between measure where the trajectory falls on the complexity spectrum.

**Eligibility.** PE is structurally computable from as few as one ordinal window, but statistically unreliable with too few. The instrument enforces a minimum of \( 2 \times D! \) eligible (non-tied) windows before reporting a PE value. This is a policy threshold --- stricter than the mathematical minimum --- chosen so that every possible pattern has room to appear at least twice.

**Current parameters.** We use \( D = 3 \), \( \tau = 1 \). This gives \( 3! = 6 \) possible ordinal patterns and requires sequences of at least 14 eligible windows. \( D = 3 \) is the minimum dimension that captures non-trivial temporal structure (two-point patterns can only be "up" or "down"). A D-sweep to \( D = 4 \) is [planned](../research/open-problems.md) --- feasible under the \( 2 \times D! \) policy --- but \( D \geq 5 \) requires longer sequences or a relaxed eligibility threshold. The principled method for choosing embedding dimension is the False Nearest Neighbors (FNN) algorithm (Kennel, Brown & Abarbanel, 1992), which identifies when adding a dimension no longer "unfolds" previously overlapping attractor regions; the D-sweep is an empirical analog for our short sequences.

## The connection: SAD trajectories as delay-coordinate observables

Here is how the pieces fit together.

The model's internal state during generation is high-dimensional and unobservable. At each generation step, the SAD instrument captures post-RoPE Q/K/V tensors from each attention head, computes both softmax and linear attention in fp32, and measures the cosine divergence between them. This produces one scalar per (layer, head) pair per step --- a time series.

Under the Takens framing, this scalar cosine divergence is the *observation function* \( h \), and the generation steps are the discrete dynamics \( \varphi \). The delay vectors formed from consecutive divergence values reconstruct the attractor of the dynamical process governing that head's attention behavior. PE on those delay vectors measures the *complexity of the reconstructed attractor* --- how many distinct temporal patterns the head visits during generation.

The first-differencing step removes the [position confound](../research/pilot-findings.md#position-confound-confirmed-and-addressed) (SAD deltas climb mechanically with prefix length due to the linear attention denominator). PE on the first-differenced trajectory measures the complexity of how the divergence *changes*, not its level.

This is what the pilot measured. The result --- [338/1024 heads with |Cohen's d| > 0.5](../research/pilot-findings.md#per-head-pe-on-first-differenced-sad-is-the-strongest-signal), correct generations showing more complex dynamics than incorrect --- is consistent with the Takens framing but does not prove it. [Gate 3](../instrument/gate-discipline.md) is designed to test whether per-head PE actually tracks attractor complexity using synthetic processes with known dynamical structure.

## Belief state geometry (Shai et al., [arXiv:2405.15943](https://arxiv.org/abs/2405.15943))

*Status: Established (external). Connection to SAD is theoretically motivated, not yet empirically tested.*

Shai, Marzen, Teixeira, Gietelink Oldenziel, and Riechers (2024) show that transformers trained on next-token prediction construct *belief state geometry* in their residual streams --- geometric structure that encodes the model's probabilistic beliefs about the hidden states of the data-generating process. This connects to SAD through computational mechanics.

**The key finding.** When a transformer is trained on sequences generated by a hidden Markov model (HMM), the activations in its residual stream linearly encode the *mixed-state presentation* (MSP) of the process --- the meta-dynamics of Bayesian belief updating over hidden states. A linear 2D projection of the final-layer residual stream recapitulates the theoretical geometry predicted by computational mechanics.

**Unifilar vs. non-unifilar.** A hidden Markov model is *unifilar* if, given the current hidden state and the observed symbol, the next hidden state is uniquely determined. Non-unifilar processes lack this property: the same (state, observation) pair can lead to multiple possible next states. This distinction determines the geometry:

- **Unifilar processes** produce point-like belief states. Each observation history maps to a single hidden state, so the belief state is always a delta distribution. The MSP geometry is discrete --- a finite set of points.
- **Non-unifilar processes** produce fractal belief states. Because observations do not uniquely determine the hidden state, Bayesian updating repeatedly splits and folds the probability distribution. Each new observation applies a contraction mapping to the belief simplex, and iterated application of these contractions produces self-similar structure --- a fractal. The more ambiguous the state transitions, the richer the fractal. For specific HMM families, Shai et al. demonstrate Sierpinski-triangle-like belief state attractors.

**Why this matters for SAD.** If transformers linearly encode belief state geometry, and that geometry can be fractal, then a smooth generic observable of the residual stream should carry information about the fractal structure. Per-head cosine divergence between softmax and linear attention is such an observable --- it is a smooth function of the Q/K/V tensors, which are linear projections of the residual stream. Under the Takens framing, PE on the resulting trajectory measures the complexity of the reconstructed attractor, which should correlate with the fractal dimension of the underlying belief state geometry.

This gives Gate 3 its design logic. Synthetic HMMs with known unifilarity properties have *computable* fractal dimensions. If per-head PE tracks fractal dimension across a family of such processes, the instrument is measuring what the theory says it should measure. If it does not, either the observable is not generic (it is blind to the relevant structure) or the theoretical connection does not hold.

**What this does not mean.** The Shai et al. result establishes that belief state geometry exists in transformer residual streams for synthetic processes. It does not establish that SAD's specific observable (softmax-linear cosine divergence) is sensitive to that geometry, that the connection survives for natural language (where the "generating process" is far more complex than any HMM), or that PE is the right summary statistic for the fractal dimension. Each of these is a separate empirical question. Gate 3 tests the first. The others are [open problems](../research/open-problems.md).

---

**References**

- Takens, F. (1981). Detecting strange attractors in turbulence. In *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics, vol. 898. Springer-Verlag, pp. 366--381.
- Mane, R. (1981). On the dimension of the compact invariant sets of certain nonlinear maps. In *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics, vol. 898. Springer-Verlag, pp. 230--242.
- Kennel, M. B., Brown, R., & Abarbanel, H. D. I. (1992). Determining embedding dimension for phase-space reconstruction using a geometrical construction. *Physical Review A*, 45(6), 3403--3411.
- Sauer, T., Yorke, J. A., & Casdagli, M. (1991). Embedology. *Journal of Statistical Physics*, 65(3--4), 579--616. (Generalizes Takens and Mane to attractors with box-counting dimension \( d_A \), requiring \( D > 2d_A \).)
- Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. *Physical Review Letters*, 88(17), 174102.
- Rosso, O. A., Larrondo, H. A., Martin, M. T., Plastino, A., & Fuentes, M. A. (2007). Distinguishing Noise from Chaos. *Physical Review Letters*, 99(15), 154102. (Introduces the complexity-entropy plane used in the planned [Renyi fingerprint](../research/open-problems.md).)
- Shai, A. S., Marzen, S. E., Teixeira, L., Gietelink Oldenziel, A., & Riechers, P. M. (2024). Transformers Represent Belief State Geometry in their Residual Stream. In *Advances in Neural Information Processing Systems* (NeurIPS 2024). [arXiv:2405.15943](https://arxiv.org/abs/2405.15943). [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html).
