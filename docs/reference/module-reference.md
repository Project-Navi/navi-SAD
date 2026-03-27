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

<!-- Phase 2: Expand with key function signatures, cross-references to theory pages -->
