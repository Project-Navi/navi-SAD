# Pilot Findings

*Last updated: 2026-03-27*

*Status: Observed in pilot. 40 samples, characterization only --- not inferential.*

---

## We hypothesized that grand-mean SAD would separate correct from incorrect generations

The naive hypothesis was straightforward: if a model is confabulating, its internal dynamics should look different from when it is generating correctly. Specifically, the cosine divergence between softmax and linear attention --- averaged across all layers and heads --- should be measurably higher (or lower) for incorrect generations than for correct ones.

This would have been convenient. A single scalar per generation, thresholded, done.

## Method

We ran a 40-sample pilot on TruthfulQA using Mistral-7B-Instruct-v0.2 (fp16, eager attention, revision-pinned, cache-off). The instrument had already passed [Gates 0--2](../instrument/gate-discipline.md): non-interference (bit-identical tokens), parity (cosine >= 0.999996, relative L2 <= 0.002759 on 2240 records), and stability (zero VRAM creep over 50 consecutive generations).

**Sample selection:** 40 rows drawn from TruthfulQA `generation` split with `random.Random(seed=42)`. Indices persisted and burned for future work.

**Labeling:** Three independent reviewers assigned labels (correct / incorrect / ambiguous) via majority vote. 92% of labels were unanimous. The 40 indices produced 28 correct, 9 incorrect, 3 ambiguous. Ambiguous samples were excluded from all group comparisons.

**Shadow scorer:** We built a deliberate string-level scorer (`truthfulqa_exact_v1`) as a shadow --- boundary-aware prefix matching against TruthfulQA's answer lists. This was designed to be simple and known-imperfect, evaluated against the human labels rather than trusted for unsupervised use.

**Generation:** Greedy decoding (`do_sample=False`), `max_new_tokens=256`, natural EOS. Single-sequence (B=1). Deterministic CUDA controls matching gate fixtures. Per-step per-layer per-head SAD deltas captured and persisted.

## The hypothesis was wrong

**Grand-mean SAD does not separate groups.** *Falsified.* The gap between correct and incorrect grand-mean SAD is 0.006 on a ~0.30 baseline. This is noise. The "SAD detects confabulation as a scalar" idea is dead.

Pooled [permutation entropy (PE)](../theory/takens-embedding.md) tells the same story: ~0.98 for both groups, negligible gap. Averaging away the per-head structure destroys whatever signal exists.

## But we found something better

When we stopped averaging and looked at individual (layer, head) pairs, structure appeared.

### Per-(layer, head) mean delta has structure

*Observed in pilot.* Using the leading-answer-span measurement boundary, 294 out of 1024 heads show |Cohen's d| > 0.5 (the conventional "medium" threshold; Cohen, 1988) between correct and incorrect groups. This threshold is not calibrated to this pilot --- different thresholds produce different head counts. Late layers (roughly L24--L31) flip sign relative to early and mid layers --- the direction of the SAD divergence reverses with depth.

This is the full-generation mean delta, not PE. It says that specific heads behave differently when the model generates correctly versus incorrectly, and that this difference has spatial structure across the layer axis.

### Per-head PE on first-differenced SAD is the strongest signal

*Observed in pilot.* This is the result worth pursuing.

For each (layer, head) pair, we extracted the SAD delta time series across generation steps, applied first-differencing to remove the position trend, and computed [permutation entropy](../theory/takens-embedding.md) (Bandt-Pompe, D=3, tau=1) on the resulting trajectory. We also computed PE on raw and residual (baseline-detrended) trajectories. We segmented each trajectory into early/mid/late thirds and computed full-sequence PE.

**338 out of 1024 heads** (32 layers x 32 heads per layer) show |Cohen's d| > 0.5 across 3 or more (mode, segment) combinations. The signal is not confined to one processing mode or one temporal segment --- the same heads recur.

**Directional asymmetry: 4.6:1 positive.** Of the heads showing large effect sizes, 4.6 times as many have positive d (correct > incorrect PE) as negative. *Observed in pilot.* Correct generations have more complex attractor dynamics. Incorrect generations are more stereotyped --- lower ordinal complexity, fewer distinct temporal patterns.

**Cross-mode recurrence.** The same heads appear across raw, first-differenced, and residual modes. Top recurring heads include L10H05 (9 mode/segment combinations), L01H27 (8), L12H24 (7), L18H07 (7), L15H12 (7). *Observed in pilot.* The fact that the signal survives three different sequence transforms --- raw, differentiated, and detrended --- is harder to explain with noise than a single-mode hit.

### Position confound confirmed and addressed

*Observed in pilot.* Both groups show SAD deltas climbing from ~0.24 to ~0.40 over the course of generation. This is expected: linear attention's denominator grows with prefix length, so the divergence from softmax increases mechanically with position.

First-differencing removes this trend. The per-head PE signal persists after detrending. The position confound is real, but the signal survives the correction.

Full position-aware SAD normalization --- adjusting for the denominator growth analytically rather than by differencing --- is [deferred](open-problems.md). First-differencing is a pragmatic correction, not a principled one.

### Shadow scorer dead

*Falsified.* The automated scorer achieved 10% agreement with human reviewers. Failure modes: hedging responses classified as ambiguous when reviewers judged them correct, multi-sentence answers where the leading span missed the actual answer, and format variations the string matcher could not handle.

Manual labels (3-reviewer majority vote, 92% unanimous) are canonical for all pilot analysis. The shadow scorer is not trusted for unsupervised use.

## Hypothesis revised

**SAD is not a truth detector.** *Frozen decision --- this architectural choice will not be revisited based on pilot results.* It is a dynamical systems probe that reconstructs per-head attractor structure via [delay-coordinate embedding](../theory/takens-embedding.md).

The theoretical anchor is Shai et al. ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)): transformers construct belief state geometry in their residual streams, and that geometry can be genuinely fractal for non-unifilar inference processes. The pilot's per-head PE asymmetry is consistent with this --- correct generations requiring the model to track more complex belief states, producing richer attractor dynamics in the heads that participate in that tracking.

But consistency is not evidence. The pilot is underpowered (n=9 incorrect), exploratory (no pre-registered hypotheses), and subject to multiple-comparison concerns (1024 heads times multiple modes and segments).

## Now we need to prove it's real

[Gate 3](../instrument/gate-discipline.md) has been redesigned. The old criterion --- AUROC on TruthfulQA correct/incorrect labels --- is retired. The new criterion tests whether per-head PE tracks the computational-mechanical complexity of the inference problem directly.

**New Gate 3 design:** Build a family of synthetic HMMs with known unifilarity properties, ranging from fully unifilar (point attractor, zero fractal dimension) to maximally non-unifilar (known fractal dimension computable from process structure). Generate matched-length sequences, feed to Mistral with next-token prediction framing, capture per-head SAD trajectories, and test for rank correlation (Spearman) between per-head PE and known fractal dimension. *Planned.*

**Why synthetic first:** Shai et al. worked with synthetic processes because they could compute exact belief state geometry. Without ground truth on fractal dimension, you cannot validate that the instrument measures what it claims to measure. Natural language benchmarks come after instrument validation on known processes.

**Pass criterion:** Significant Spearman rank correlation in L15--21 heads, surviving permutation null. *Planned.*

See the [Roadmap](roadmap.md) for the full path from here to natural language benchmarks.

## What this does not mean

These findings are characterization, not evidence. Specifically:

- **338/1024 heads was small-n inflation.** The permutation null on the 40-sample data returned p=0.25 (not significant; null range [172, 768]). The 400-sample replication (282 correct, 68 incorrect) found **zero** recurring heads at |d|>0.5. The recurrence count statistic is dead on TruthfulQA.

- **4.6:1 positive asymmetry reversed at 400 samples.** The 400-sample d-landscape shows 83.4% *negative* d (incorrect PE > correct PE). The pilot's positive direction was a small-n artifact. Confound control machinery (asymmetry null, length-matched, unanimous-only) built in PR #31; not yet executed on this data.

- **Cohen's d values are exploratory, not evidential.** Every d value in the pilot was computed post-hoc on underpowered groups. Do not use them to set frozen thresholds, rank heads, or make publishable claims.

- **"Correct = complex, incorrect = stereotyped" is a description, not a mechanism.** The pilot cannot distinguish between (a) the model entering a stereotyped regime because it lacks the right answer, (b) confabulation producing mechanically simpler attention patterns, or (c) some other explanation entirely.

- **The instrument can lie.** SAD measures cosine divergence between two attention paths. If the divergence is insensitive to the dynamical structure we care about, the instrument will produce numbers that look structured but mean nothing. Gate 3 is designed to detect exactly this failure mode.

- **TruthfulQA correctness labels are one regime partition among many.** The instrument is hypothesis-agnostic. Confabulation detection is one application. The pilot used correct/incorrect labels as a convenient partition; the instrument may be more naturally suited to partitions defined by the generating process rather than by output quality.
