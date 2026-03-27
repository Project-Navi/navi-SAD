# Changelog

Research milestones and instrument changes. This is a research changelog, not a software release log.

## 2026-03-25 --- Pilot Complete, Grand Mean Dead

40-sample TruthfulQA pilot with 3-reviewer majority-vote labels (92% unanimous). Grand-mean SAD does not separate correct from incorrect (0.006 gap on ~0.30 baseline). Per-head PE on first-differenced trajectories shows structural signal: 338/1024 heads with |d|>0.5 across 3+ (mode, segment) combos. 4.6:1 directional asymmetry. Gate 3 redesigned around synthetic HMM benchmarks.

## 2026-03-24 --- Gate 2 Passes

50 consecutive generations with full instrumentation. Zero VRAM creep (0.0 MiB spread). CPU RSS growth 0.7 MiB. Provenance round-trip validated.

## 2026-03-24 --- Gates 0 and 1 Pass

Instrument verified on Mistral-7B-Instruct-v0.2 (fp16, eager, revision-pinned). Gate 0: bit-identical tokens and logits. Gate 1: cosine >= 0.999996, relative L2 <= 0.00276 across 2240 records.

## 2026-03-24 --- Milestones A + B Complete

Core math, types, I/O, mock hooks, temporal analysis. 237 CPU tests passing.
