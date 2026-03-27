# Gate 2 --- Stability

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

50 consecutive generations with full instrumentation and JSONL serialization. Zero VRAM creep (0.0 MiB spread, limit 16 MiB). CPU RSS growth 0.7 MiB (limit 128 MiB). Provenance round-trip validated.

See [Gate Discipline](gate-discipline.md) for the stability methodology and [I/O modules](../reference/module-reference.md) for the JSONL serialization pipeline.

<!-- Phase 2: Baseline methodology, warmup discipline, provenance validation details, per-head delta range checks -->
