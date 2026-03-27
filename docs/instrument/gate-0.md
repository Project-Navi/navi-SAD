# Gate 0 --- Non-interference

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

<!-- Phase 2: Full test methodology, what "bit-identical" means in practice, per-step/per-layer bijection details -->
