# The SAD Instrument

*Status: Proven by gates (instrument validation). Theoretical framing theoretically motivated (not yet empirically grounded).*

SAD captures post-RoPE Q/K/V tensors from inside the model's native [attention forward](../instrument/adapter-discipline.md), then recomputes both softmax and linear attention in fp32. The [cosine distance](../reference/glossary.md#cosine-divergence) between per-head outputs produces a scalar trajectory over generation steps --- one time series per (layer, head) pair, which we treat as a [delay-coordinate embedding](takens-embedding.md).

SAD is not a truth detector. It is a dynamical systems probe that reconstructs per-head attractor structure. What you ask about that structure is a separate question.

---

## End-to-end pipeline

### 1. Model loading

The model is loaded with `attn_implementation="eager"` (hard requirement) and native dtype (fp16 for gate verification). KV cache is disabled (`use_cache=False`). The model revision is pinned in gate fixtures to ensure reproducibility. Currently Mistral-7B-Instruct-v0.2 only --- other families earn registry entries after their gates pass.

### 2. Registry lookup and adapter installation

`get_family_config(model.config)` reads `model.config.architectures[0]` and looks up the corresponding `ModelFamilyConfig` in the registry. For Mistral, this returns a Tier A config with `adapter_factory=MistralAdapter`. The `InstrumentManager` installs the adapter on every attention layer, replacing each module's `forward` method with a [verbatim upstream copy](../instrument/adapter-discipline.md) containing capture callbacks.

### 3. Per-step capture during generation

During `model.generate()`, each forward pass fires the patched forward for every attention layer. At each layer, insertion point 1 calls the capture callback with post-RoPE `query_states`, `key_states`, `value_states`.

Step accounting is handled by a `LogitsProcessor` injected into the generation loop via `LogitsProcessorList`. The processor increments the manager's `step_idx` after each forward pass completes across all layers. Each generation step produces exactly `num_layers` records.

### 4. SAD delta computation

Inside the capture callback, for each layer at each step:

1. Q/K/V are cloned, detached, and upcast to fp32
2. GQA expansion: if `num_kv_heads != num_q_heads`, K and V are expanded via `repeat_interleave` (Mistral-7B: 8 KV heads expanded to 32)
3. Newest-token query slice: `q_last = q_fp32[:, :, -1:, :]`
4. **Softmax path**: scaled dot-product attention in fp32. \( \text{scores} = q \cdot K^T / \sqrt{d_k} \), softmax, matmul with V
5. **Linear path**: ELU+1 feature map on Q and K. Accumulated \( S = K^T V \) via einsum. Normalized by \( z = \sum K_{\text{mapped}} \)
6. **Per-head cosine distance**: \( 1 - \cos(\text{softmax}_h, \text{linear}_h) \) for each head

The result is a `StepRecord(step_idx, layer_idx, per_head_delta)` appended to the record list.

### 5. Serialization

After generation, records are packed into a `RawSampleRecord` with provenance metadata and written to gzipped JSONL. Raw records are immutable --- never modified after writing.

### 6. Downstream signal processing

Analysis operates on serialized records, never during inference:

- **Aggregation**: uniform mean across layers and heads per step, producing a per-token delta series. Raises on non-contiguous `step_idx` (fail-closed).
- **Finite differences**: first, second, third differences of the delta series.
- **[Permutation entropy](takens-embedding.md)**: per-(layer, head) PE on first-differenced SAD trajectories. Bandt-Pompe ordinal patterns (D=3, tau=1) with tie exclusion. Eligibility minimum: 2*D! points.

## Scope limitations

- **Cache-off only.** `use_cache=False` is a method definition, not a performance choice. Generalization to cache-on inference is unverified.
- **Mistral only.** Other families earn their entries after passing Gates 0 and 1.
- **Single sequence.** The instrument pipeline assumes `B=1`.
- **Eager attention only.** SDPA and Flash Attention are incompatible with the [forward-replacement adapter](../instrument/adapter-discipline.md).
- **fp16 for gates, q8 minimum for production.** Quantization below q8 introduces dequantization artifacts as a confound.
