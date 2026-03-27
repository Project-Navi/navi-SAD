---
hide:
  - navigation
  - toc
---

<div class="hero-glow" markdown>

# navi-SAD

**A dynamical systems probe for LLM inference.**

Runs softmax and linear attention in parallel on the same frozen weights, measures per-head cosine divergence, and reconstructs the model's internal attractor via delay-coordinate embedding.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[Theory](theory/sad-instrument.md){ .md-button }

</div>

---

## Gate Status

| Gate | What | Status | Last verified |
|------|------|--------|---------------|
| 0 | Non-interference --- identical tokens + logits with/without hooks | **Pass** | 2026-03-24 |
| 1 | Parity --- recomputed fp32 softmax through o_proj matches native | **Pass** | 2026-03-24 |
| 2 | Stability --- 50 generations, zero VRAM creep | **Pass** | 2026-03-24 |
| 3 | Rank correlation --- per-head PE vs known fractal dimension | In progress | --- |

## What's New

**2026-03-25** --- [Pilot findings published](research/pilot-findings.md). Grand mean dead. Per-head PE alive. Gate 3 redesigned.

**2026-03-24** --- [Gate 2 passes](instrument/gate-2.md). 50 consecutive generations, zero VRAM creep.

**2026-03-24** --- [Gates 0 and 1 pass](instrument/gate-discipline.md). Instrument verified on Mistral-7B.

---

## Documentation

| Section | Contents |
|---------|----------|
| [Getting Started](getting-started/installation.md) | Install, run the gates |
| [Theory](theory/sad-instrument.md) | What SAD measures and why --- Takens' embedding, capacity gap, adjacent literature |
| [Instrument](instrument/gate-discipline.md) | Gate discipline, validation proofs, adapter rules |
| [Research](research/pilot-findings.md) | Pilot findings, roadmap, open problems |
| [Reference](reference/module-reference.md) | Module reference, frozen decisions, glossary, changelog |
