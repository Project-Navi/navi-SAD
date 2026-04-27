# Closed Case Study: TruthfulQA Length-Matched Null

*Last updated: 2026-04-27*

*Status: Closed methodological case study. No application claim is made.*

---

The 40-sample TruthfulQA pilot and its 400-sample replication are now closed as a methodological case study. Three successive framings of the same claim --- that SAD separates correct from incorrect generations on natural-language QA --- were tested and falsified. The final null --- a length-matched permutation test on the dense-small directional asymmetry --- returned p=0.96. The directional pattern observed in the d-landscape was a generation-length confound, not signal.

The repository's framing has pivoted. SAD is a runtime measurement instrument; the [empirical validation runs through Gate 3](../instrument/gate-discipline.md) on synthetic HMM benchmarks with known fractal dimensions, not through TruthfulQA. No application claim --- including confabulation, hallucination, truthfulness, or correctness detection --- is asserted by this repository pre-Gate-3.

This page exists so that readers can follow what was tried, what was killed, and why the natural-language thread is closed.

---

## Method

We ran a 40-sample pilot on TruthfulQA using Mistral-7B-Instruct-v0.2 (fp16, eager attention, revision-pinned, cache-off). The instrument had already passed [Gates 0--2](../instrument/gate-discipline.md): non-interference (bit-identical tokens), parity (cosine >= 0.999996, relative L2 <= 0.002759 on 2240 records), and stability (zero VRAM creep over 50 consecutive generations).

**Sample selection (40 + 400).** Indices drawn from TruthfulQA `generation` split with `random.Random(seed=42)`. Indices persisted. The 400-sample replication used the same selection scheme on a larger draw.

**Labeling.** Three independent reviewers per sample, majority vote. 40-sample: 92% unanimous, 28 correct / 9 incorrect / 3 ambiguous. 400-sample (3 Opus reviewers): 88.5% unanimous, 282 correct / 68 incorrect / 50 ambiguous. Ambiguous samples were excluded from all group comparisons.

**Shadow scorer.** A deliberate string-level scorer (`truthfulqa_exact_v1`) was built as a known-imperfect bootstrap, evaluated against human labels rather than trusted for unsupervised use.

**Generation.** Greedy decoding (`do_sample=False`), `max_new_tokens=256`, natural EOS. Single-sequence (B=1). Deterministic CUDA controls matching gate fixtures. Per-step per-layer per-head SAD deltas captured and persisted.

**Confound controls (PR #31).** Three nulls executed on the 400-sample data: head-level signed asymmetry null (stratified by token-count bins, two-sided primary), length-matched analysis with pair-restricted null (greedy nearest-neighbor on token count), and unanimous-only label robustness check. All three operate on the per-head mean d across present (mode, segment) combos; baseline deviation reported as diagnostic.

---

## Three framings, three falsifications

### 1. Grand-mean SAD separates groups --- falsified

The naive hypothesis was that the cosine divergence between softmax and linear attention --- averaged across all layers and heads --- should be measurably different for incorrect generations than for correct ones. The 40-sample data show a 0.006 gap on a ~0.30 baseline. This is noise. Pooled [permutation entropy (PE)](../theory/takens-embedding.md) shows the same: ~0.98 for both groups, negligible gap. Averaging away the per-head structure destroys whatever signal might exist.

### 2. Per-head PE recurrence count at |d|>0.5 --- falsified

The 40-sample analysis surfaced 338/1024 heads with |Cohen's d| > 0.5 across 3+ (mode, segment) combinations on first-differenced trajectories, with a 4.6:1 directional asymmetry favoring positive d (correct PE > incorrect PE) and apparent cross-mode recurrence. The stratified permutation null on this count returned p=0.25 (not significant; null range [172, 768]). The 400-sample replication --- with 282 correct and 68 incorrect samples --- found **zero** recurring heads at |d|>0.5. The original count was small-n inflation at n=9 incorrect.

### 3. 400-sample dense-small directional asymmetry --- falsified by length-matched null

The 400-sample d-landscape showed a dense-small-effect regime: max |d|=0.58, mean |d|=0.134, with 83.4% *negative* d (incorrect PE > correct PE). The direction had reversed from the pilot's 4.6:1 positive. With confound controls executed:

- The pair-restricted permutation null on the length-matched subset returned **p=0.96**. The dense-small negative asymmetry did not survive matching on token count; the direction was a generation-length confound, not signal.
- The head-level signed asymmetry null (stratified across the full cohort) and the unanimous-only robustness check were also run. With the matched-design null at p=0.96, the directional pattern on TruthfulQA is closed as an observed regularity with no inferential support.

The shadow scorer's agreement with human reviewers was 10% at 40 samples and 18.5% at 400 samples; it is not fit for unsupervised use. Manual majority-vote labels were canonical throughout.

---

## Position confound

Both correct and incorrect groups in the pilot showed SAD deltas climbing from ~0.24 to ~0.40 over the course of generation. This is mechanical, not surprising: linear attention's normalizer \( z = \sum_j \phi(K_j) \) grows with prefix length while softmax does not, so cosine divergence between the two outputs increases with position by construction. First-differencing the per-head delta series removes the trend empirically, which is why the per-head PE analyses operated on first-differenced trajectories.

The remaining issue is that *between-group* differences in generation length are not removed by first-differencing if the same underlying confound shapes how ordinal patterns accumulate over short vs. long sequences. The 400-sample data show a length disparity between correct and incorrect groups that is sufficient to drive the observed dense-small directional asymmetry. The pair-restricted permutation null on the length-matched subset (p=0.96) is the design that controls for this directly. A principled, position-aware analytical normalization remains an [open problem](open-problems.md).

## What this case study shows

- **An interesting-looking head-level pattern is not evidence.** The 40-sample 338/1024 count was visually striking and survived multiple sequence transforms (raw / first-differenced / residual). It still failed its permutation null at scale.
- **Direction reversal between pilot and replication is a warning, not a signal.** The pilot's 4.6:1 positive direction reversed to 83.4% negative at 400 samples. Reversal under more data is what you expect when the pilot was sampling noise.
- **Generation-length is a load-bearing confound on natural-language QA.** Linear attention's denominator grows with prefix length; SAD deltas climb mechanically with position. The length-matched null is the design that controls for this directly. p=0.96 indicates the dense-small directional asymmetry is fully explained by generation-length differences between correct and incorrect samples in this dataset.
- **The instrument worked. The hypothesis did not.** Gates 0--2 passed throughout. The instrument measured what it claimed to measure (parity to fp32 softmax, no interference with native inference, no memory drift). What was falsified was the chain of reasoning from per-head SAD trajectories to TruthfulQA correctness labels --- not the instrument itself.

---

## Why no application claim is asserted pre-Gate-3

This case study did not produce evidence that per-head SAD trajectories separate correct from incorrect natural-language generation on TruthfulQA. It also did not produce evidence against the broader Takens / belief-state-geometry framing --- it tested a specific dataset and a specific labelling scheme, both of which contain confounds that the instrument is not designed to disentangle on its own.

The repository asserts no application claim --- confabulation, hallucination, truthfulness, or correctness detection --- because none has been validated. The TruthfulQA application thread is closed and is not on the roadmap.

## What replaced this thread: Gate 3

[Gate 3](../instrument/gate-discipline.md) has been redesigned around synthetic HMMs with known unifilarity properties, ranging from fully unifilar (point attractor, zero fractal dimension) to maximally non-unifilar (known fractal dimension computable from the generating-process structure). Matched-length sequences are fed to Mistral; per-head SAD trajectories are captured; per-head PE (across D=3..7) is correlated against known fractal dimension via Spearman rank.

**Why synthetic first.** Shai et al. ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)) worked with synthetic processes precisely because they could compute exact belief-state geometry. Without ground truth on the quantity the instrument is supposed to measure, you cannot validate the instrument. Natural-language regimes --- if they are explored at all post-Gate-3 --- come after instrument validation on processes whose dynamical structure is known by construction.

**Pass criterion.** Significant Spearman rank correlation in L15--21 heads, surviving permutation null. *Planned.*

See the [Roadmap](roadmap.md) for the full path from here.

---

## Artifacts (gitignored)

The TruthfulQA pilot and replication artifacts remain on disk for reproducibility, but are not the basis of any current claim:

- `results/pilot_gate3/` --- 40-sample run (samples, review, recurrence null)
- `results/pilot_gate3_400/` --- 400-sample replication (samples, review with per-reviewer batches, recurrence null, confound controls including length-matched null at p=0.96, full report)

These are gitignored. Indices and seeds are persisted; the case study is reproducible end-to-end. It is not, however, evidence for any application of SAD.
