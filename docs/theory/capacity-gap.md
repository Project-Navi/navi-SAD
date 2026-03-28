# Capacity Gap (Han et al.)

*Status: Established (external). Han et al. 2024, arXiv:2412.06590, NeurIPS 2024.*

Han et al. prove that softmax attention is *injective* --- different queries produce different attention distributions --- while linear attention is *not* --- distinct queries can collapse to identical outputs. This capacity gap is the structural basis for using softmax-linear divergence as a diagnostic signal.

SAD does not claim that divergence directly measures truth. It measures how much the model relies on its full nonlinear attention capacity versus operating in a regime where the weaker linear mechanism suffices.

---

## The core result: injectivity separates the two mechanisms

The standard attention mechanisms can be written as functions that map a query vector to an attention weight distribution over key-value pairs. Han et al. formalize this as:

\[
S_K(Q_i) = \left[\frac{\exp(Q_i^\top K_1)}{\sum_j \exp(Q_i^\top K_j)},\;\ldots,\;\frac{\exp(Q_i^\top K_N)}{\sum_j \exp(Q_i^\top K_j)}\right]
\]

for softmax attention, and:

\[
L_K(Q_i) = \left[\frac{\phi(Q_i)^\top \phi(K_1)}{\sum_j \phi(Q_i)^\top \phi(K_j)},\;\ldots,\;\frac{\phi(Q_i)^\top \phi(K_N)}{\sum_j \phi(Q_i)^\top \phi(K_j)}\right]
\]

for linear attention, where \( \phi \) is a feature map (kernel function) applied element-wise.

Both functions map from query space \( \mathbb{R}^d \) to the probability simplex over \( N \) keys. The question is whether different queries produce different distributions.

## What "injective" means here

A function \( f: A \to B \) is injective if and only if for all \( x, y \in A \) with \( x \neq y \), it holds that \( f(x) \neq f(y) \). No two distinct inputs produce the same output.

Applied to attention: an injective attention function guarantees that every distinct query vector produces a unique attention weight distribution. If two queries attend to the key-value pairs differently, the mechanism can distinguish them. If the function is not injective, distinct queries can receive identical attention weights --- the mechanism cannot tell them apart.

Han et al. call this *semantic confusion*: when different queries (carrying different meaning) collapse to the same attention pattern, the head loses the ability to differentiate the contexts those queries represent.

## Softmax is injective

**Proposition 1** (Han et al., 2024). Given keys \( K \in \mathbb{R}^{N \times d} \) with \( \mathrm{rank}(K) = d \) and \( \mathrm{rank}([K,\; \mathbf{1}_{N \times 1}]) = d + 1 \), for all \( p, q \in \mathbb{R}^d \) with \( p \neq q \):

\[
S_K(p) \neq S_K(q)
\]

The proof proceeds by contradiction. Suppose \( S_K(p) = S_K(q) \). The softmax normalization means:

\[
\frac{\exp(p^\top K_i)}{\sum_j \exp(p^\top K_j)} = \frac{\exp(q^\top K_i)}{\sum_j \exp(q^\top K_j)} \quad \forall\; i
\]

This implies \( \exp((p - q)^\top K_i) \) is constant across all \( i \), which under the rank conditions forces \( p = q \), contradicting the assumption.

The rank conditions are mild. \( \mathrm{rank}(K) = d \) means the keys span the query space (not degenerate). \( \mathrm{rank}([K, \mathbf{1}]) = d + 1 \) means the keys are not all translations of each other (the constant vector is not in the key span). Both conditions hold generically for trained models.

The exponential function is doing the work. It maps linear differences to multiplicative ratios, and the resulting normalization preserves enough information to reconstruct the original query direction.

## Linear attention is not injective

**Proposition 2** (Han et al., 2024). For any continuous feature map \( \phi: \mathbb{R}^d \to \mathbb{R}^d \), there exist \( p, q \in \mathbb{R}^d \) with \( p \neq q \) such that:

\[
L_K(p) = L_K(q)
\]

The failure mode is concrete. Consider the ReLU feature map \( \phi(x) = \mathrm{ReLU}(x) \). For any positive scalar \( \alpha \neq 1 \):

\[
L_K(\alpha \cdot p) = L_K(p)
\]

because the normalization in the denominator cancels the scaling factor. All collinear queries with the same direction --- regardless of magnitude --- produce identical attention weights. The linear attention mechanism cannot distinguish a "loud" query from a "quiet" one pointing in the same direction.

This is not limited to scaling. Han et al. show that with a learned affine feature map \( \phi(x) = \mathrm{ReLU}(Ax + b) \), four queries with *different directions and magnitudes* can produce exactly the same attention distribution. The collapse is structural, not a corner case.

## Why this makes the cosine divergence informative

The capacity gap has a direct consequence for SAD. At each generation step, the instrument computes both softmax and linear attention on the same Q/K/V tensors and measures the cosine distance between the per-head outputs. The injectivity gap means:

**When linear attention suffices.** If the model is operating in a regime where the query-key relationships are simple enough that linear attention can replicate what softmax does, the two outputs will be similar and cosine divergence will be low. The head is not using the full nonlinear capacity of softmax --- the exponential's ability to sharply separate and distinguish queries is not needed.

**When the model needs full nonlinear capacity.** If the model is in a regime where different queries must produce meaningfully different attention patterns --- where the distinctions softmax makes are load-bearing --- linear attention will collapse some of those distinctions. The outputs diverge, and cosine distance increases. The head is relying on capabilities that linear attention structurally cannot replicate.

The divergence is not random. It tracks the *functional gap* between what the two mechanisms can represent. Heads that need to make fine-grained distinctions among queries (high injectivity demand) will show high divergence. Heads doing coarse operations (where many queries can safely collapse) will show low divergence.

Over the course of generation, this produces a *trajectory* --- a time series of divergence values, one per step, per (layer, head) pair. Changes in that trajectory reflect changes in how much nonlinear capacity the head is using. [Permutation entropy](takens-embedding.md) on that trajectory measures how complex those changes are --- how many distinct temporal patterns of capacity usage the head visits during generation.

## What this does not mean

The capacity gap result is a structural property of the attention mechanisms. It says something precise about what linear attention *cannot do*. It does not, by itself, tell you what any particular divergence value *means* for the model's output quality.

**High divergence does not mean hallucination.** It means the model is using attention capabilities that linear attention cannot replicate. A model answering a difficult factual question correctly may show high divergence because the task requires fine-grained query discrimination. A model confidently producing wrong output may show high or low divergence depending on the internal dynamics.

**Low divergence does not mean correctness.** It means the model is operating in a regime where linear attention suffices. This could be because the task is simple, because the head is doing a coarse operation, or because the model is in a stereotyped generation mode that happens to not require sharp query distinctions.

**The divergence is a measurement, not a diagnosis.** SAD measures the gap. What you conclude from the gap depends on additional analysis --- the [per-head PE structure](../research/pilot-findings.md), the layer and head identity, the comparison to [known dynamical baselines](../research/roadmap.md). The capacity gap result tells you *why* the measurement is informative; it does not tell you what any particular value means.

The [pilot findings](../research/pilot-findings.md) show that the interesting signal is not in the divergence values themselves (grand-mean SAD does not separate groups), but in the *temporal complexity* of the divergence trajectories (per-head PE on first-differenced SAD). The capacity gap explains why the trajectories carry information: they reflect time-varying demands on the attention mechanism's nonlinear capacity.

---

Note: the formulation above omits the scaling factor \( \sqrt{d_k} \) from the standard softmax presentation (Vaswani et al., 2017). The scaling does not affect the injectivity argument. The ELU+1 feature map \( \phi(x) = \text{elu}(x) + 1 \) used in the navi-SAD instrument is the specific choice introduced by Katharopoulos et al. (2020); the normalizer \( z = \sum_j \phi(K_j) \) grows linearly with sequence length, which is the source of the [position confound](../research/pilot-findings.md#position-confound-confirmed-and-addressed) in SAD trajectories.

**References**

- Han, D., Ye, Y., Xia, Z., Han, Y., Pan, X., Li, X., Lu, J., Song, S., & Huang, G. (2024). Bridging the Divide: Reconsidering Softmax and Linear Attention. In *Advances in Neural Information Processing Systems* (NeurIPS 2024). [arXiv:2412.06590](https://arxiv.org/abs/2412.06590).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is All You Need. In *Advances in Neural Information Processing Systems* (NeurIPS 2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In *ICML 2020*. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236).
