# Gate 0 --- Non-interference

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

The adapter produces bit-identical tokens and logits under deterministic greedy decoding with and without instrumentation installed. Per-step/per-layer record bijection verified across 32 layers. The observer does not perturb the system.

See [Gate Discipline](gate-discipline.md) for the full methodology and [Adapter Discipline](adapter-discipline.md) for the non-interference invariant.

---

## What "bit-identical" means

Under deterministic greedy decoding (`do_sample=False`, `use_cache=False`), with deterministic CUDA controls (`torch.backends.cudnn.deterministic=True`, `benchmark=False`, seeded RNG), the model produces the same output tokens on every run. Gate 0 first verifies this baseline determinism: two consecutive greedy runs without instrumentation must produce `torch.equal` token sequences.

With that baseline established, Gate 0 compares:

1. **Token identity:** Generate with and without the adapter installed. Output token sequences must satisfy `torch.equal` --- not "close", not "within tolerance", identical tensors. Tested across three prompt types at 30 tokens each.

2. **Logit identity:** A single forward pass with and without instrumentation. Full logit tensors compared via `torch.testing.assert_close(rtol=0, atol=0)`. Token identity alone is insufficient because two different logit distributions can produce the same greedy-decoded token. Logit identity proves the adapter genuinely does not perturb the computation, not just that it gets lucky on argmax.

## Per-step/per-layer record bijection

Every generation step must produce exactly `num_layers` StepRecords, one per layer, with no duplicates and no gaps. For a generation of `max_new_tokens` steps across 32 layers, the total record count must be exactly `32 * max_new_tokens`.

The test enforces this structurally:

- Group records by `step_idx`. The set of step indices must equal \( \{0, 1, \ldots, \text{max\_new\_tokens} - 1\} \).
- Within each step, the set of `layer_idx` values must equal \( \{0, 1, \ldots, 31\} \).
- Within each step, the record count must equal 32 (catches duplicates a set check alone would miss).
- Each record's `per_head_delta` must have length 32 (one per attention head).

## Why this matters

The non-interference invariant is the foundation of the entire instrument. If installing capture hooks changes the model's behavior --- even by a single logit value --- then the SAD deltas we measure are deltas of a *different system* than the uninstrumented model. Every downstream measurement depends on Gate 0 passing. The [adapter](adapter-discipline.md) achieves this by being a verbatim copy of the upstream forward with three observation-only insertions that read tensors but never write to them.
