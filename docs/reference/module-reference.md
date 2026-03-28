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

## `navi_sad.analysis`

| Module | Responsibility |
|--------|---------------|
| `types.py` | Frozen dataclasses: EligibilityCell/Table, PermutationNullConfig, RecurrenceStatistic/Profile, PermutationNullResult, RecurrenceNullReport, DLandscape, AsymmetryStatistic, SubsetSpec, MatchingDiagnostics, SelectionDiagnostics, AsymmetryNullResult, BaselineDeviation |
| `loader.py` | Boundary: load + validate review/samples integrity + parse per-step to StepRecord + load per-reviewer votes |
| `prep.py` | Two-layer prep: prepare_series_data() (D-independent) + compute_pe_bundle() (D-dependent). Subset prep + baseline deviation diagnostic. |
| `eligibility.py` | Per-class x mode x segment accounting. No statistics. |
| `recurrence.py` | compute_d_matrix(), recurrence_from_d_matrix(), summarize_d_matrix() -> DLandscape, compute_head_asymmetry() -> AsymmetryStatistic. Numpy vectorized Cohen's d. |
| `permutation.py` | Stratified label permutation, Phipson-Smyth p-values, asymmetry null (stratified + pair-restricted). RNG confined here only. |
| `matching.py` | Greedy nearest-neighbor length matching. Deterministic, no RNG. |
| `selection.py` | Deterministic cohort selection (unanimous-only filter). |
| `report.py` | Provenance building, markdown rendering for recurrence null and confound controls reports |

### Key functions

- `compute_d_matrix(lookup, labels, num_layers, num_heads)` --- full Cohen's d matrix, never discarded. Returns `DMatrix`.
- `summarize_d_matrix(d_matrix, num_layers, num_heads)` --- distribution stats, directional counts, threshold sweep. Returns `DLandscape`.
- `compute_head_asymmetry(d_matrix, num_layers, num_heads)` --- per-head mean-d, sign classification, min-combo gating. Returns `AsymmetryStatistic`.
- `run_permutation_null(lookup, labels, token_counts, config, num_layers, num_heads)` --- stratified permutation null for recurrence count.
- `run_asymmetry_null(lookup, labels, token_counts, ...)` --- stratified permutation null for signed asymmetry. Returns `AsymmetryNullResult`.
- `run_paired_asymmetry_null(lookup, labels, pairs, ...)` --- pair-restricted null for matched designs.
- `prepare_series_data(results_dir, num_layers, num_heads)` --- load, validate, extract series, compute baseline. D-independent.
- `prepare_series_data_from_subset(data, indices, baseline, ...)` --- in-memory subset prep with provided baseline.
- `compute_pe_bundle(series_data, pe_config)` --- compute PE features at a specific D.
- `compute_baseline_deviation(subset_head_series, full_baseline)` --- subset-vs-full baseline diagnostic.
- `match_by_token_count(labels, token_counts)` --- greedy nearest-neighbor matching with pairs.
- `select_unanimous(reviewer_votes, majority_labels)` --- unanimous-only cohort filter.
- `load_reviewer_votes(labeling_dir)` --- reads per-reviewer batch files.

## `navi_sad.stats`

| Module | Responsibility |
|--------|---------------|
| `effect_size.py` | Shared Cohen's d with validity guards (GuardedStat). POOLED_VAR_EPS numeric guard. |

## Not yet ported

| Module | Language | Responsibility |
|--------|----------|---------------|
| `navi_dsc_renyi.h` | C++ (production kernel, not in this repo) | Renyi entropy, Renyi complexity (Jensen-Renyi divergence from uniform), and parametric fingerprint curves. See [Open Problem 4](../research/open-problems.md). |
