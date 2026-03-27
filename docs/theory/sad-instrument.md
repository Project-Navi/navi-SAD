# The SAD Instrument

*Status: Proven by gates (instrument validation). Theoretical framing theoretically motivated (not yet empirically grounded).*

SAD captures post-RoPE Q/K/V tensors from inside the model's native [attention forward](../instrument/adapter-discipline.md), then recomputes both softmax and linear attention in fp32. The [cosine distance](../reference/glossary.md#cosine-divergence) between per-head outputs produces a scalar trajectory over generation steps --- one time series per (layer, head) pair, which we treat as a [delay-coordinate embedding](takens-embedding.md).

SAD is not a truth detector. It is a dynamical systems probe that reconstructs per-head attractor structure. What you ask about that structure is a separate question.

<!-- Phase 2: Full content from README Method section, instrument architecture diagram, the "what the instrument can see" framing -->
