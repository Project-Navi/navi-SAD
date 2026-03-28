# Changelog

Research milestones and instrument changes. This is a research changelog, not a software release log.

## 2026-03-28 --- Confound Controls Machinery (PR #31)

Signed asymmetry null (head-level, two-sided primary), length-matched analysis (pair-restricted null), unanimous-only analysis. New modules: matching.py, selection.py. Baseline deviation diagnostic. 401 CPU tests. Code built and tested; not yet run on 400-sample data.

## 2026-03-28 --- 400-Sample Replication, Recurrence Count Dead

400-sample TruthfulQA run with 3-Opus-reviewer majority-vote labels (88.5% unanimous). 282 correct, 68 incorrect, 50 ambiguous. Recurrence null at |d|>0.5: zero recurring heads. The 40-sample pilot's 342-head count was small-n inflation. D-landscape: max |d|=0.58, mean |d|=0.134, 83.4% negative (incorrect PE > correct PE). Direction reversed from pilot. Observed pattern, not validated result --- asymmetry null machinery built in PR #31, not yet run.

## 2026-03-27 --- Analysis Module + Infrastructure (PRs #28, #29)

PR #28: PE recurrence null analysis module. Types, eligibility, recurrence, permutation, loader, report. Shared stats extraction (Cohen's d to stats/effect_size.py). 5 audit rounds. 309 CPU tests.

PR #29: D-matrix persistence and numpy vectorization. compute_d_matrix(), summarize_d_matrix(), DLandscape type. StepRecord boundary parsing. Prep module (prepare_series_data + compute_pe_bundle). Full-grid denominator accounting. 345 CPU tests.

## 2026-03-25 --- Pilot Complete, Grand Mean Dead

40-sample TruthfulQA pilot with 3-reviewer majority-vote labels (92% unanimous). Grand-mean SAD does not separate correct from incorrect (0.006 gap on ~0.30 baseline). Per-head PE on first-differenced trajectories shows structural signal: 338/1024 heads with |d|>0.5 across 3+ (mode, segment) combos. 4.6:1 directional asymmetry. Shadow scorer dead (10% agreement with human reviewers). Gate 3 redesigned around synthetic HMM benchmarks.

## 2026-03-24 --- Gate 2 Passes

50 consecutive generations with full instrumentation. Zero VRAM creep (0.0 MiB spread). CPU RSS growth 0.7 MiB. Provenance round-trip validated.

## 2026-03-24 --- Gates 0 and 1 Pass

Instrument verified on Mistral-7B-Instruct-v0.2 (fp16, eager, revision-pinned). Gate 0: bit-identical tokens and logits. Gate 1: cosine >= 0.999996, relative L2 <= 0.002759 across 2240 records.

## 2026-03-24 --- Milestones A + B Complete

Core math, types, I/O, mock hooks, temporal analysis. 237 CPU tests passing.
