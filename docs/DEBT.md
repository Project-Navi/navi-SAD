# Technical Debt

Items identified during the multivector audit of the analysis module (2026-03-27).
Each item has a source audit, severity, and context for why it matters.

## Empirical Rigor

### GQA non-independence (CRITICAL for publication)

**Source:** Empirical rigor audit, finding #5

Mistral-7B uses 8 KV groups with 4 Q heads per group. The 32 heads within each KV group share K/V tensors, so their Cohen's d values are correlated. The recurrence count treats all 1024 heads as independent counting units, which inflates the raw count.

The permutation null partially compensates (same correlation structure in both observed and null), but a second-order interaction between GQA correlation and true signal could bias the p-value in an unknown direction.

**Required before publication:**
- KV-group-collapsed analysis: max-d or majority-vote recurrence per KV group (256 effective heads instead of 1024)
- Compare p-values between head-level and group-level analyses to bound the GQA effect

**Caveat currently documented in:** `scripts/pe_recurrence_null.py` markdown output, `docs/research/pilot-findings.md`

### Cohen's d small-sample bias (IMPORTANT for publication)

**Source:** Empirical rigor audit, finding #1

With n=9 incorrect, Cohen's d has a positive bias (~2.2%). Hedges' g correction factor `J = 1 - 3/(4*(n_a+n_b-2) - 1)` would reduce d by ~2.2%. The bias partially cancels in the permutation test (same estimator used for observed and null), but cancellation is imperfect when eligibility censoring makes effective sample sizes unequal across (mode, segment) combos.

**Required before publication:**
- Switch `compute_cohens_d` to Hedges' g (or add `compute_hedges_g` alongside)
- Re-run recurrence analysis and compare head counts

### d_threshold sensitivity sweep (IMPORTANT for publication)

**Source:** Empirical rigor audit, finding #6

The 338-head count at d_threshold=0.5 could be highly sensitive to threshold choice. The profile addresses the min_combos dimension but not the d_threshold dimension.

**Required before publication:**
- Sensitivity heatmap: d_threshold (0.3, 0.4, 0.5, 0.6, 0.8) x min_combos (1..12) -> recurring head count and p-value
- Report whether p-value remains stable across threshold choices

### Stratification effective permutation count (SUGGESTION)

**Source:** Empirical rigor audit, finding #3

With n=9 incorrect and 2-bin stratification, some strata may have as few as 4-5 incorrect samples, reducing the effective number of distinct permutations well below 10,000. The Phipson-Smyth formula handles discrete distributions correctly, but the resolution is limited.

**Action:**
- Add a diagnostic that computes and logs the number of distinct permutations per stratum
- Warn if any stratum has fewer than 1000 effective distinct permutations

## Architecture

### Inverted dependency: analysis -> pilot (SUGGESTION)

**Source:** Types audit, finding I3

`analysis/recurrence.py` imports `compute_cohens_d` from `pilot/helpers.py`. The analysis module is core instrument code; the pilot module is pilot-quality code. The dependency direction is inverted.

**Action:** Extract `compute_cohens_d` (and future `compute_hedges_g`) into a shared statistics utility module (e.g., `navi_sad/stats/effect_size.py`) that both `pilot` and `analysis` can import from. Combine with the Hedges' g work above.

### PELookup type alias vs NewType (SUGGESTION)

**Source:** Types audit, finding T2

`PELookup` is a plain `TypeAlias`. Any dict with the same structural shape passes mypy. `NewType` would force explicit wrapping at construction sites.

**Action:** Consider `NewType("PELookup", dict[...])` if the type is used by external callers. Currently only `build_pe_lookup` constructs it, so the risk is low.

### No from_dict() for serialized types (SUGGESTION)

**Source:** Types audit, finding T3; Footguns audit, finding #3

`RecurrenceStatistic.to_dict()` converts tuple keys to `"layer,head"` strings. `RecurrenceProfile.to_dict()` converts int keys to str keys. All three conversions are one-way with no documented inverse. If programmatic consumers need to deserialize JSON reports back into typed objects, they must reverse-engineer the string formats.

**Action:** Add `from_dict()` classmethods when programmatic consumption becomes a requirement. Document the serialization format in docstrings.

## Hardening

### Frozen dataclass mutable fields (CAUTION)

**Source:** Types audit, finding D1; Footguns audit, finding #1

`frozen=True` prevents attribute reassignment but not in-place mutation of list/dict fields. Documented on `RecurrenceNullReport` but not on the other five affected dataclasses.

**Action:** Consider `tuple` for `bin_boundaries` and `null_counts`. Consider `types.MappingProxyType` wrappers in `__post_init__` for dict fields if true immutability becomes important. Currently convention-only.

### to_dict() shallow copies mutable fields (CAUTION)

**Source:** Footguns audit, additional finding

`RecurrenceNullReport.to_dict()` now copies `bin_boundaries` and `bin_counts` (fixed in audit commit). But `RecurrenceStatistic.to_dict()` creates new string keys for `per_head_combo_counts` (safe — new dict), and `PermutationNullResult.to_dict()` creates new string keys for `null_percentiles` (safe). No remaining shallow-copy issues.

### Eligibility vs PE-present conceptual coupling (CAUTION)

**Source:** Footguns audit, finding #4

The distinction between "eligible" (passed length threshold) and "pe_present" (eligible AND pe != None) exists only in the `EligibilityCell` docstring and the `build_pe_lookup` filter. It is not enforced by the type system. A developer adding a new code path that feeds differently-filtered data into `compute_combo_cohens_d` would get wrong results silently.

**Action:** Consider a `PEPresentLookup` newtype or a runtime assertion in `compute_combo_cohens_d` that verifies all PE values are non-None.
