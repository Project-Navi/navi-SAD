# Gate 0 --- Non-interference

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

See [Gate Discipline](gate-discipline.md) for the full methodology and [Adapter Discipline](adapter-discipline.md) for the non-interference invariant.

<!-- Phase 2: Full test methodology, what "bit-identical" means in practice, per-step/per-layer bijection details -->
