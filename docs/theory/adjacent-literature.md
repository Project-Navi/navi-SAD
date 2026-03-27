# Adjacent Literature

*Status: Survey of related work. Connections to SAD are theoretical or analogical. Last updated 2026-03-27.*

To our knowledge, no published method runs two attention mechanisms in parallel on the same frozen weights as a dynamical systems probe. SAD combines known ingredients (linear attention, cosine divergence, delay-coordinate embedding via ordinal patterns) in a new configuration. This page surveys the adjacent landscape so a reader can locate SAD within it.

The approaches below fall into roughly three categories: **representation decomposition** (what features exist in the model's internals), **causal tracing** (which components matter for a given output), and **runtime monitoring** (what signals are available during generation). SAD belongs to the third category but differs from all published runtime monitors in its dynamical systems framing --- it reconstructs attractor geometry rather than computing scalar diagnostics.

---

## 1. Sparse Autoencoders (SAEs)

**What they do.** SAEs train an autoencoder with a sparsity penalty on activations from a frozen LLM, decomposing residual stream or MLP outputs into a dictionary of sparse, ideally interpretable "features." The goal is to recover the model's internal concepts as individual dictionary elements. OpenAI scaled SAEs to GPT-4 class models (Gao et al., 2024); Google DeepMind released Gemma Scope, the largest open SAE infrastructure.

**The features debate.** Whether the learned dictionary elements correspond to genuine computational primitives of the model remains contested. DeepMind's mechanistic interpretability team reported in March 2025 that SAEs underperformed dense linear probes on safety-relevant downstream tasks --- specifically, detecting harmful user intent and generalizing to novel jailbreak attempts. SAE probes showed substantially worse out-of-distribution generalization than simple linear probes trained on the same residual stream activations. Even SAEs fine-tuned on chat-specific data only closed about half the performance gap. The team concluded that "SAEs in their current form are far from achieving" the goal of capturing a canonical set of true concepts, citing feature absorption, noisy representations, and high false-negative rates on interpretable latents. They announced they were deprioritizing fundamental SAE research, stating they "do not think SAEs will be a game-changer for interpretability" and speculating "the interpretability community is somewhat over-invested in SAEs" ([DeepMind Safety Research, 2025](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)). A subsequent paper (arXiv:2506.23845) argued that the comparative advantage of SAEs lies in *discovering unknown* concepts rather than acting on known ones --- a reframing that narrows the claimed use case substantially.

**Limitation.** SAEs decompose a snapshot of model state at a single position. They have no temporal dimension --- they cannot characterize how a feature evolves across generation steps. No ground truth exists for whether the learned features are "real" computational primitives or artifacts of the training objective.

**How SAD differs.** SAD does not decompose representations into features. It tracks the *trajectory* of a scalar observable (cosine divergence between softmax and linear attention) across generation steps, per (layer, head). The signal is temporal dynamics, not static feature activation. The two approaches are orthogonal: SAE features describe what the model represents; SAD trajectories describe how the model's dynamics evolve.

---

## 2. Circuit Tracing / Attribution Graphs (Anthropic, 2025)

**What it does.** Anthropic's circuit tracing replaces a model's MLPs with cross-layer transcoders to build "attribution graphs" --- directed graphs where nodes are sparse features and edges represent causal influence between them for a specific prompt. The method traces the chain of intermediate steps from input features through hidden features to the output logit, providing per-prompt mechanistic explanations. Published in March 2025 on Claude 3.5 Haiku ([Anthropic, 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); companion paper: [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)).

**The coverage limitation.** The authors state that attribution graphs "provide us with satisfying insight for about a quarter of the prompts we've tried." Each graph is an execution trace for a single prompt --- it shows how the model computes a specific output, not how it computes outputs in general. The method cannot explain how attention patterns are formed (QK-circuits are frozen during analysis), which the authors note "prevents us from understanding a variety of behaviors that hinge on the model 'fetching' a piece of information from earlier in the context." Graph completeness averages 0.80 (80% of important inputs explained by features rather than error nodes); graph replacement scores average 0.61.

**The NP-hardness result.** Independently, Adolfi, Vilas, & Wareham (ICLR 2025 Spotlight, [arXiv:2410.08025](https://arxiv.org/abs/2410.08025)) proved that many circuit discovery queries are NP-hard and remain fixed-parameter intractable relative to model and circuit features. The problems are inapproximable under additive, multiplicative, and probabilistic approximation schemes. This establishes a theoretical ceiling on what circuit-finding methods can guarantee.

**Limitation.** Per-prompt analysis that does not generalize to algorithmic understanding. Requires building a replacement model (transcoders). Works on approximately 25% of prompts. The NP-hardness result means that scaling circuit discovery to larger models or more complex behaviors faces fundamental computational barriers, not just engineering ones.

**How SAD differs.** SAD requires no transcoders or replacement models. It works on the original frozen model and produces per-step, per-(layer, head) signals across the entire generation. It captures temporal dynamics --- how the model's behavior evolves --- rather than static causal paths for a single output token.

---

## 3. EigenTrack (Ettori et al., 2025)

**What it does.** EigenTrack ([arXiv:2509.15735](https://arxiv.org/abs/2509.15735)) constructs sliding-window activation matrices from hidden states during generation, extracts covariance-spectrum statistics (leading eigenvalues, spectral gaps, spectral entropy, KL divergence from the Marchenko-Pastur random matrix baseline), and feeds these into a lightweight recurrent classifier. The classifier tracks temporal shifts in representation structure, signaling hallucination and OOD drift. The authors report AUROC ranges across model families (LLaMA, Qwen, Mistral, LLaVa) and benchmarks, operating on a single forward pass without resampling.

**Limitation.** Requires training a supervised recurrent classifier on labeled hallucination/OOD data. The covariance-spectrum statistics are computed over hidden state activations aggregated across positions --- they do not provide per-head resolution. The classifier is model- and task-specific; generalization to new models or task distributions requires retraining.

**How SAD differs.** SAD provides per-(layer, head) trajectories, not aggregated spectral statistics. Permutation entropy on delay-coordinate embeddings is a classifier-free complexity measure --- no supervised training is needed. The theoretical framing is different: EigenTrack asks "does the covariance spectrum look anomalous?" while SAD asks "what is the attractor geometry of this head's dynamics?" Both use spectral properties during generation; they look at different objects (hidden state covariance vs. attention divergence) at different granularities (aggregate vs. per-head).

---

## 4. D2HScore (Ding et al., 2025)

**What it does.** D2HScore ([arXiv:2509.11569](https://arxiv.org/abs/2509.11569)) decomposes hallucination signals into two dimensions: *intra-layer dispersion* (the mean L2 distance of token embeddings from their layer centroid, measuring semantic breadth within a layer) and *inter-layer drift* (the L2 distance of attention-selected key token representations across adjacent layers, measuring semantic depth across the model's depth). The final hallucination score is the normalized sum of both components. It is training-free and label-free.

**Limitation.** D2HScore measures semantic properties of token representations (how spread out, how much they change across layers), not temporal dynamics across generation steps. It operates on the completed generation --- dispersion and drift are computed over the full sequence of token embeddings, making it a post-hoc analysis rather than a real-time, per-step signal. The dispersion measure is sensitive to sequence length and prompt complexity.

**How SAD differs.** SAD produces a per-step signal during generation: each generation step yields a new SAD delta per (layer, head). The temporal trajectory *is* the measurement, not a summary statistic over the final state. SAD measures dynamical complexity (attractor geometry via PE), not semantic coherence (centroid distances). D2HScore's two-dimensional decomposition is elegant but static; SAD's per-head trajectories are high-dimensional and temporal.

---

## 5. Semantic Entropy (Farquhar et al., 2024)

**What it does.** Semantic entropy (Farquhar, Kossen, Kuhn, & Gal, [Nature, 2024](https://www.nature.com/articles/s41586-024-07421-0); earlier version at ICLR 2023) estimates uncertainty at the level of meaning rather than token sequences. For each input, the model generates M (typically 5--10) independent completions. An NLI model (e.g., DeBERTa) clusters completions by semantic equivalence --- answers that mean the same thing are grouped together. Entropy is then computed over the cluster distribution. High semantic entropy indicates genuine uncertainty; low semantic entropy with a wrong answer indicates confabulation.

**The cost problem and the probe shortcut.** The method requires 5--10 full forward passes per input, making it impractical for real-time inference. Semantic Entropy Probes (Kossen et al., [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)) address this by training linear probes on hidden states to approximate semantic entropy from a single forward pass, but this reintroduces the supervised training requirement and model specificity that the original method avoided.

**Limitation.** The full method is expensive (5--10x compute). The probe shortcut requires labeled semantic entropy scores for training and is model-specific. Both versions operate at the sequence level, not the token or step level, and provide no information about *which* internal components are uncertain or *how* uncertainty evolves during generation.

**How SAD differs.** SAD requires a single forward pass with parallel attention branches. The per-step, per-(layer, head) signal reveals the temporal structure of the model's dynamics --- not just whether the model is uncertain, but how the dynamics of individual attention heads evolve across the generation. SAD does not require multiple generations, NLI models, or supervised probes.

---

## 6. Probing Classifiers / Linear Probes

**What they do.** Probing classifiers train lightweight models (typically linear) on hidden state activations to predict properties of interest --- truthfulness, factual accuracy, sentiment, syntactic structure. If a linear probe achieves high accuracy, the property is likely linearly separable in the representation space.

**The truthfulness geometry.** Burns et al. ([arXiv:2212.03827](https://arxiv.org/abs/2212.03827)) introduced Contrast-Consistent Search (CCS), an unsupervised method that finds truth directions in activation space by enforcing logical consistency (a statement and its negation should have opposite truth values). Li et al. ([arXiv:2306.03341](https://arxiv.org/abs/2306.03341)) developed Inference-Time Intervention (ITI), which identifies attention heads that separate true from false statements and shifts activations along truthful directions at inference time, improving TruthfulQA accuracy from 32.5% to 65.1% on Alpaca. Marks & Tegmark (COLM 2024, [arXiv:2310.06824](https://arxiv.org/abs/2310.06824)) showed that truth/falsehood has emergent linear structure in LLM representations, with simple difference-in-mean probes generalizing across datasets.

**The middle-layer finding.** Probing accuracy for truthfulness-related properties peaks in middle layers (e.g., layers 12--18 in a 32-layer model), not in the final layer. Semantic Entropy Probes similarly capture uncertainty best from mid-to-late layers. This suggests that truth-relevant representations are constructed progressively and may be partially overwritten by later processing.

**Limitation.** Probes require labeled training data and are model- and task-specific. Marks et al. noted that learned truthfulness representations degrade under superficial changes in input presentation --- the probes may detect surface correlates rather than deep truth representations. Generalization across datasets and models remains fragile.

**How SAD differs.** SAD is unsupervised --- no labeled data, no training. The signal comes from attention dynamics (softmax-linear divergence), not hidden state activations. Where probes ask "is this representation truthful?" SAD asks "what is the dynamical complexity of this head's trajectory?" The two approaches measure different things: probes measure static representational geometry; SAD measures temporal attractor structure.

---

## 7. Neural Uncertainty Principle (2026)

**What it does.** The Neural Uncertainty Principle (NUP, [arXiv:2603.19562](https://arxiv.org/abs/2603.19562)) proposes that the input and its loss gradient are conjugate observables subject to an irreducible uncertainty bound --- analogous to the position-momentum uncertainty principle in quantum mechanics. In language models, weak prompt-gradient coupling leaves generation under-constrained, creating hallucination risk. In vision models, the same principle manifests as adversarial fragility. The paper introduces a single-backward probe that captures input-gradient correlation at the prefill stage, detecting hallucination risk before any tokens are generated.

**Limitation.** The principle applies "in near-bound regimes," which may constrain its applicability to specific operating conditions. The single-backward probe requires a backward pass at prefill time --- not available in all deployment settings. The theoretical framework unifies adversarial fragility and hallucination under one principle, but the practical detection method has not yet been validated at scale across diverse tasks and models.

**How SAD differs.** NUP measures a static property of the prompt (gradient coupling) before generation begins. SAD measures temporal dynamics *during* generation. NUP answers "is this prompt likely to produce hallucination?" SAD answers "what is the model's dynamical state evolving into during this generation?" The two are complementary: NUP is a pre-generation risk signal; SAD is a per-step trajectory characterization.

---

## 8. Verbal Uncertainty Mismatch (Ji et al., 2025)

**What it does.** Ji et al. ([arXiv:2503.14477](https://arxiv.org/abs/2503.14477)) report that "verbal uncertainty" --- how confidently a model expresses itself --- is governed by a single linear feature in the model's representation space, and that this feature has only moderate correlation with the model's actual semantic uncertainty. The mismatch between high semantic uncertainty and low verbal uncertainty (i.e., the model is unsure but speaks assertively) is a better predictor of hallucination than semantic uncertainty alone. By intervening on the verbal uncertainty feature at inference time, they reduce confident hallucinations by approximately 30%.

**Limitation.** Requires computing semantic uncertainty (which itself requires multiple generations or a trained probe) to detect the mismatch. The intervention modifies model behavior rather than just observing it. The verbal uncertainty feature is identified per model --- it is not guaranteed to transfer across architectures.

**How SAD differs.** SAD is purely observational --- it does not intervene on the model's representations. It does not require computing semantic uncertainty as a prerequisite. The signal is structural (attractor geometry) rather than calibration-based (uncertainty mismatch). Where the verbal uncertainty work asks "is the model saying it's confident when it shouldn't be?" SAD asks "what does the trajectory of this head's dynamics look like?" --- a question that does not require a ground-truth uncertainty estimate.

---

## 9. What Is Novel About SAD

To our knowledge, no published method runs dual attention mechanisms in parallel on the same frozen weights as a dynamical systems probe. The individual components are all known:

- **Linear attention** is a well-studied approximation (Katharopoulos et al., 2020; Han et al., [arXiv:2412.06590](https://arxiv.org/abs/2412.06590), NeurIPS 2024).
- **Cosine divergence** between representations is standard.
- **Delay-coordinate embedding** via ordinal patterns is the canonical tool for attractor reconstruction in nonlinear dynamics (Takens, 1981; Bandt & Pompe, 2002).

The configuration is new. SAD uses the [capacity gap](capacity-gap.md) between softmax and linear attention (Han et al. prove softmax is injective while linear attention is not) as a *diagnostic observable* of the model's internal dynamics, then applies [Takens' embedding](takens-embedding.md) to reconstruct the attractor geometry of per-head trajectories across generation steps. Permutation entropy on these reconstructions measures the complexity of the attractor --- not as a generic heuristic, but as the measurement that Bandt-Pompe ordinal patterns were designed for.

The [theoretical anchor](../research/three-repo-unification.md) is Shai et al. (NeurIPS 2024, [arXiv:2405.15943](https://arxiv.org/abs/2405.15943)), who showed that transformers trained on HMM sequences construct belief state geometry in their residual streams and that this geometry can be genuinely fractal for non-unifilar inference processes. If per-head attention is a deterministic function of the residual stream, and the residual stream encodes belief state geometry, then per-head SAD divergence should carry information about belief state complexity. [Gate 3](../instrument/gate-discipline.md) tests this prediction with synthetic HMM benchmarks where ground-truth fractal dimensions are known.

---

## The Gap SAD Fills

The field's approaches to understanding and monitoring LLM internals cluster around two paradigms. The first is **static decomposition**: SAEs decompose activations into features; probing classifiers train supervised detectors on hidden states; the verbal uncertainty work identifies linear directions in representation space. These methods analyze a snapshot of model state at a single position or over a completed generation. They answer "what does the model represent?" but not "how do those representations evolve during generation?" The second is **causal tracing**: circuit tracing maps attribution paths from input to output; activation patching identifies sufficient components. These methods ask "which components matter for this output?" but produce per-prompt execution traces, not temporal characterizations of the generative process.

Neither paradigm captures **temporal dynamics during generation with per-(layer, head) resolution**. EigenTrack comes closest --- it tracks spectral statistics of hidden activations over time --- but operates at aggregate granularity and requires a trained classifier. D2HScore captures cross-layer structure but is post-hoc. Semantic entropy characterizes uncertainty over completed generations, not the step-by-step evolution of internal state. The Neural Uncertainty Principle measures a static property of the prompt before generation begins. Every existing approach either ignores the temporal dimension, aggregates away the per-head resolution, or requires supervision.

SAD occupies this gap. It treats each attention head's softmax-linear divergence as a scalar time series over generation steps, applies delay-coordinate embedding to reconstruct the attractor geometry of that head's dynamics, and measures the complexity of the reconstruction via permutation entropy. The instrument is grounded in attractor reconstruction theory (Takens, 1981), not in ad hoc feature engineering. The per-(layer, head) resolution means the measurement preserves the architectural structure of the model --- each head's trajectory is an independent observable. The Shai et al. result provides the theoretical prediction that this observable should carry information about the computational-mechanical complexity of the inference problem. Whether it does is an open empirical question. Gate 3 will test it. The pilot data (40 samples, characterization only) show structural signal in per-head PE on first-differenced trajectories --- 338/1024 heads with |d|>0.5 across 3+ (mode, segment) combinations --- but this is characterization, not evidence. The gap between "structural signal observed" and "signal corresponds to belief state complexity" is exactly what the synthetic HMM benchmark is designed to close.

---

## References

| Work | Citation | Key Contribution |
|------|----------|-----------------|
| SAEs (scaling) | Gao et al., 2024 ([OpenAI](https://cdn.openai.com/papers/sparse-autoencoders.pdf)) | Scaled SAEs to frontier models |
| SAEs (deprioritization) | DeepMind Safety, 2025 ([Medium](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)) | SAEs underperform linear probes on downstream safety tasks |
| SAE survey | [arXiv:2503.05613](https://arxiv.org/abs/2503.05613) | Comprehensive survey of SAE methods and applications |
| Circuit tracing | Anthropic, 2025 ([methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) | Attribution graphs on Claude 3.5 Haiku |
| Circuit complexity | Adolfi, Vilas, & Wareham, ICLR 2025 Spotlight ([arXiv:2410.08025](https://arxiv.org/abs/2410.08025)) | NP-hardness of circuit discovery queries |
| EigenTrack | Ettori et al., 2025 ([arXiv:2509.15735](https://arxiv.org/abs/2509.15735)) | Spectral activation tracking for hallucination/OOD |
| D2HScore | Ding et al., 2025 ([arXiv:2509.11569](https://arxiv.org/abs/2509.11569)) | Dispersion + drift hallucination score |
| Semantic entropy | Farquhar et al., Nature 2024 ([doi](https://www.nature.com/articles/s41586-024-07421-0)) | Meaning-level uncertainty estimation |
| Semantic entropy probes | Kossen et al., 2024 ([arXiv:2406.15927](https://arxiv.org/abs/2406.15927)) | Linear probes approximating semantic entropy |
| CCS | Burns et al., 2023 ([arXiv:2212.03827](https://arxiv.org/abs/2212.03827)) | Unsupervised truth direction discovery |
| ITI | Li et al., 2023 ([arXiv:2306.03341](https://arxiv.org/abs/2306.03341)) | Inference-time truthfulness intervention |
| Geometry of truth | Marks & Tegmark, COLM 2024 ([arXiv:2310.06824](https://arxiv.org/abs/2310.06824)) | Linear truth structure in LLM representations |
| NUP | 2026 ([arXiv:2603.19562](https://arxiv.org/abs/2603.19562)) | Prompt-gradient uncertainty bound |
| Verbal uncertainty | Ji et al., [arXiv:2503.14477](https://arxiv.org/abs/2503.14477) | Semantic-verbal uncertainty mismatch |
| Capacity gap | Han et al., NeurIPS 2024 ([arXiv:2412.06590](https://arxiv.org/abs/2412.06590)) | Softmax injective, linear attention not |
| Belief state geometry | Shai et al., NeurIPS 2024 ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)) | Fractal belief state structure in residual streams |
| Mech interp outlook | Nanda, 80,000 Hours podcast, Sept 2025 ([link](https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/)) | "The most ambitious vision... is probably dead" |
