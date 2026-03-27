# Pilot Findings

*Status: Observed in pilot. 40 samples, characterization only --- not inferential.*

*Last updated: 2026-03-25*

The 40-sample TruthfulQA pilot falsified the naive hypothesis and produced one result worth pursuing.

**Grand-mean [SAD](../theory/sad-instrument.md) does not separate groups.** 0.006 gap on ~0.30 baseline. Dead.

**Per-head [permutation entropy (PE)](../theory/takens-embedding.md) on first-differenced SAD is the strongest signal.** 338/1024 heads show |d|>0.5 across 3+ (mode, segment) combinations. Directional asymmetry: 4.6:1 positive (correct = more complex dynamics, incorrect = more stereotyped). Cross-mode recurrence across raw, diff, and residual.

This shows structural signal in per-head PE. It does not show that this signal corresponds to belief state complexity. [Gate 3](../instrument/gate-discipline.md) tests that prediction.

**Shadow scorer dead.** 10% agreement with human reviewers. Manual labels (3-reviewer majority vote, 92% unanimous) are canonical.

<!-- Phase 2: Full narrative (hypothesis -> method -> what died -> what survived -> what's next), position confound analysis, per-layer sign flip, Cohen's d matrices (exploratory) -->
