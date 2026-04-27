---
name: adapter-upstream-diff
description: Diff src/navi_sad/core/adapter.py's patched MistralAttention.forward against the installed transformers upstream source, masking the three SAD INSERTION blocks. Use when auditing adapter drift, bumping transformers, or after any edit to core/adapter.py.
disable-model-invocation: true
---

# Adapter Upstream Diff

Verify that `core/adapter.py`'s patched forward stays faithful to upstream `MistralAttention.forward`, modulo a small, explicit set of deliberate departures.

## Why this exists

CLAUDE.md describes the patched forward as a *verbatim copy* of `transformers.models.mistral.modeling_mistral.MistralAttention.forward`. In practice, "verbatim" means **logic-identical** with a documented set of cosmetic and eager-only departures. Any change **outside** that documented set is potential drift that silently breaks Gate 0 — it cannot be caught by mypy or ruff.

This skill surfaces the textual diff between the two forward bodies after stripping the insertion blocks and other known-acceptable departures. Remaining diff output is the signal: a reviewer matches it against the **Known deliberate departures** list below and anything that doesn't match is treated as drift.

## The three allowed insertions

1. `=== SAD INSERTION 1: Tier A capture post-RoPE Q/K/V ===` ... `=== END INSERTION 1 ===`
2. `=== SAD INSERTION 2: parity capture (Gate 1 only) ===` ... `=== END INSERTION 2 ===`
3. `=== SAD INSERTION 3: pre-o_proj diagnostic (parity only) ===` ... `=== END INSERTION 3 ===`

The helper strips these before diffing, so they never appear in the output.

## Known deliberate departures (expected in the diff output)

Treat any of the following as **acceptable** — not drift:

1. **Closure capture**: upstream takes `self`, the patched forward is a closure that captures `module` from its parent function. The helper normalises `self` → `module` on both sides.
2. **Eager-only enforcement**: upstream dispatches via `ALL_ATTENTION_FUNCTIONS[...]`; patched hard-fails with `RuntimeError` if `_attn_implementation != "eager"` and calls `eager_attention_forward` directly. This is the enforcement mechanism for the eager-only frozen decision.
3. **Type annotation stripping**: patched drops the `Cache | None` annotation on `past_key_values` (to avoid an unused `Cache` import), and drops `Unpack[FlashAttentionKwargs]` from `**kwargs`.
4. **ruff reformatting** of long lines: multi-arg function calls and long dict literals get split across lines. No logical change.
5. **Dropped comment**: the `# main diff with Llama` trailing comment on `sliding_window=` is not carried into the patched version.
6. **`@deprecate_kwarg` decorator**: upstream's decorator is dropped because closures cannot carry decorators. Logic-equivalent since the kwarg is aliased the same way at call sites.

If the diff output matches only items 1–6, the adapter is correct. Anything else is drift.

## When to run

- After any edit to `src/navi_sad/core/adapter.py`.
- Before and after a `transformers` version bump.
- As part of the pre-merge audit for adapter-touching PRs.
- When Gate 0 fails unexpectedly.

## Procedure

1. Confirm `transformers` is installed at the pinned version:

   ```bash
   uv run python -c "import transformers; print(transformers.__version__)"
   ```

   Must be in `[4.57.0, 4.58.0)`.

2. Run the diff helper:

   ```bash
   uv run python .claude/skills/adapter-upstream-diff/diff_adapter.py
   ```

3. Inspect output:
   - **Clean** (exit 0) — the normalised bodies are identical. Extremely unlikely given the deliberate departures documented above; exit 0 means the adapter and upstream are perfectly aligned on every non-insertion character.
   - **Drift output** (exit 1) — expected. Scan each diff hunk against the **Known deliberate departures** list. For every hunk:
     - If it matches a documented departure → acceptable.
     - If it doesn't match → this is new drift. Decide whether the change is:
       - An accidental edit → revert.
       - A legitimate upstream change → bump `_COMPAT_MIN` / `_COMPAT_MAX` in `core/adapter.py`, copy the new upstream body, update the departure list in this SKILL.md, then re-run `gate-check`.
       - A new required insertion → add it as `SAD INSERTION 4` with matching END marker, update this skill's insertion list, and re-calibrate Gate 1.

## What the helper does

`diff_adapter.py`:

- Loads the installed `transformers.models.mistral.modeling_mistral` source via `inspect.getsource(MistralAttention.forward)`.
- Loads `core/adapter.py` and extracts the nested `forward` function (defined inside `MistralAdapter._make_capturing_forward`) via AST.
- Normalises both sides before diffing:
  - strips the three `SAD INSERTION N` ... `END INSERTION N` blocks
  - drops decorator lines (upstream's `@deprecate_kwarg`)
  - drops `# type: ignore[...]` suffix comments (patched side)
  - converts `Optional[X]` to `X | None` (upstream uses `typing.Optional`, patched uses PEP 604)
  - drops `**kwargs: Unpack[FlashAttentionKwargs]` annotation
  - normalises `self` → `module` on both sides (the allowed closure substitution)
  - strips per-line leading / trailing whitespace — logic matters, indentation is enforced by the parser
- Prints a unified diff of what remains. Exits 1 if any diff exists; exit 0 means truly identical after normalisation.

## References

- `CLAUDE.md` § "Adapter discipline", "Milestone C"
- `src/navi_sad/core/adapter.py` top-level docstring
- `_COMPAT_MIN` / `_COMPAT_MAX` constants in `core/adapter.py`
- Upstream source: `transformers==4.57.x`, `transformers/models/mistral/modeling_mistral.py`
