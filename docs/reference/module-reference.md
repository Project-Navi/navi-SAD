# Module Reference

*Status: Reference. Hand-maintained --- this is a research harness, not a library with a stable API.*

## `navi_sad.core`

| Module | Responsibility |
|--------|---------------|
| `spectral.py` | Softmax + linear attention (newest-token), GQA expansion, cosine distance |
| `hooks.py` | Mock hook manager with GQA-correct reshape |
| `adapter.py` | MistralAdapter: Tier A forward-replacement with 3 marked insertions |
| `instrument.py` | InstrumentManager: orchestrates adapters, step accounting, SAD delta computation |
| `registry.py` | Mistral-only model family registry |
| `types.py` | StepRecord, RawSampleRecord, ModelFamilyConfig, ParityConfig, ParityRecord |

## `navi_sad.signal`

| Module | Responsibility |
|--------|---------------|
| `ordinal.py` | Bandt-Pompe ordinal patterns with tie exclusion, PE (D=3). Generic engine, not SAD-specific. |
| `pe_features.py` | SAD-specific PE wrapper: per-(layer, head) extraction, first-differencing, detrending, segmentation, eligibility |
| `derivatives.py` | Finite differences on delta series |
| `aggregation.py` | Uniform-mean aggregation, fail-closed on step_idx gaps |
| `types.py` | OrdinalResult, DerivedSampleRecord dataclasses |

## `navi_sad.pilot`

| Module | Responsibility |
|--------|---------------|
| `schema.py` | Typed write-side schema for pilot artifacts |
| `helpers.py` | Extraction, shadow scorer, scalar computation, alignment, integrity validation |

## `navi_sad.io`

| Module | Responsibility |
|--------|---------------|
| `writer.py` | Gzipped JSONL writer (raw records) |
| `reader.py` | Gzipped JSONL reader |
| `derived.py` | Derive analysis records from raw |

## Key functions

### `core/spectral.py`

- `softmax_attention_last_token(q_last, k, v)` --- fp32 scaled dot-product attention for the newest token. Returns \( [B, H, 1, D] \). See [The SAD Instrument](../theory/sad-instrument.md).
- `linear_attention_last_token(q_last, k, v)` --- ELU+1 feature map, accumulated \( S = K^T V \) via einsum, normalized by \( z = \sum K_{\text{mapped}} \). Returns \( [B, H, 1, D] \).
- `per_head_cosine_distance(softmax_out, linear_out)` --- \( 1 - \cos(\text{softmax}_h, \text{linear}_h) \) for each head. Returns \( [H] \).
- `expand_kv_heads(kv, num_q_heads)` --- GQA expansion via `repeat_interleave`.

### `core/adapter.py`

- `MistralAdapter.install(attn_module, capture_fn, parity_fn)` --- replaces the module's `forward` with a [verbatim upstream copy](../instrument/adapter-discipline.md) containing capture callbacks. Version guard on transformers `~=4.57`.

### `signal/ordinal.py`

- `ordinal_patterns(x, D, tau)` --- extracts Bandt-Pompe ordinal patterns with tie exclusion. Returns `OrdinalResult(patterns, counts, ties_excluded)`. See [Takens' Embedding](../theory/takens-embedding.md).
- `permutation_entropy(x, D, tau)` --- Shannon entropy of the ordinal pattern distribution. Normalized to \( [0, 1] \).

### `signal/pe_features.py`

- `extract_pe_features(records, D, tau)` --- SAD-specific PE wrapper. Per-(layer, head) extraction, first-differencing, detrending, segmentation, eligibility gating (minimum 2*D! points).
