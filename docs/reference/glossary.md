# Glossary

Terms used across the navi-SAD documentation. Definitions are project-specific where noted.

**Attractor**
:   A set of states toward which a dynamical system evolves. In navi-SAD, the per-head SAD trajectory is treated as a delay-coordinate reconstruction of the model's internal attractor.

**Bandt-Pompe ordinal patterns**
:   The rank-order permutations of consecutive values in a time series, introduced by Bandt & Pompe (2002). For embedding dimension D, there are D! possible patterns.

**Belief state geometry**
:   The geometric structure that transformers construct in their residual streams to represent probabilistic beliefs about hidden states. See Shai et al. (NeurIPS 2024).

**Cosine divergence**
:   The cosine distance (1 - cosine similarity) between softmax and linear attention outputs for the same head. The core SAD measurement.

**Delay-coordinate embedding**
:   Reconstructing a dynamical system's state space from a single scalar time series by plotting each value against its time-delayed predecessors. Justified by Takens' embedding theorem (1981).

**Epsilon-machine**
:   The minimal unifilar hidden Markov model that generates a given process. From computational mechanics (Crutchfield & Young, 1989).

**GQA (Grouped Query Attention)**
:   An attention variant where multiple query heads share key-value heads. Mistral-7B uses GQA (32 query heads, 8 KV heads). SAD expands KV heads to match query heads before computing divergence.

**Non-unifilar**
:   A hidden Markov model where the current observation does not uniquely determine the hidden state transition. Non-unifilar processes have fractal belief state geometry.

**Permutation entropy (PE)**
:   The Shannon entropy of the ordinal pattern distribution in a delay-coordinate embedding. Measures the complexity of the trajectory's temporal structure.

**SAD (Spectral Attention Divergence)**
:   The per-head cosine distance between softmax and linear attention outputs, computed at each generation step. The core observable of the navi-SAD instrument.

**Unifilar**
:   A hidden Markov model where each (state, observation) pair uniquely determines the next state. Unifilar processes have point-like (non-fractal) belief state geometry.
