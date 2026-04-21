#!/usr/bin/env python3
"""Diff core/adapter.py's patched forward against upstream transformers.

Exit codes:
  0 — normalised bodies are identical (no remaining diff). Rare in practice
      because of the known deliberate departures (eager-only dispatch, closure
      signature, dropped type annotations). See SKILL.md.
  1 — normalised bodies differ. Expected in the steady state; scan each hunk
      against SKILL.md's "Known deliberate departures" list and treat anything
      unmatched as drift.
  2 — extraction or import failure (adapter.py structure changed, or
      transformers is missing / wrong version).

Usage:
    uv run python .claude/skills/adapter-upstream-diff/diff_adapter.py
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = REPO_ROOT / "src" / "navi_sad" / "core" / "adapter.py"

# Strip from BEGIN to matching END marker, inclusive of both lines.
INSERTION_BLOCK_RE = re.compile(
    r"^[ \t]*#[ \t]*=+[ \t]*SAD INSERTION[ \t]+(\d+).*?"
    r"^[ \t]*#[ \t]*=+[ \t]*END INSERTION[ \t]+\1[ \t]*=*[ \t]*\n",
    re.MULTILINE | re.DOTALL,
)
SELF_RE = re.compile(r"\bself\b")
TYPE_IGNORE_RE = re.compile(r"[ \t]*#[ \t]*type:[ \t]*ignore(?:\[[^\]]+\])?")
DECORATOR_RE = re.compile(r"^[ \t]*@[A-Za-z_][\w.]*.*\n", re.MULTILINE)
OPTIONAL_RE = re.compile(r"Optional\[([^\]]+)\]")


def extract_patched_forward_source(adapter_path: Path) -> str:
    """Return the source of the nested ``forward`` inside ``_make_capturing_forward``."""
    source = adapter_path.read_text()
    tree = ast.parse(source)
    for outer in ast.walk(tree):
        if (
            isinstance(outer, ast.FunctionDef | ast.AsyncFunctionDef)
            and outer.name == "_make_capturing_forward"
        ):
            for inner in ast.walk(outer):
                if (
                    isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef)
                    and inner.name == "forward"
                ):
                    segment = ast.get_source_segment(source, inner)
                    if segment is None:
                        raise RuntimeError(
                            "ast.get_source_segment returned None for nested forward"
                        )
                    return segment
    raise RuntimeError(
        f"nested 'forward' inside _make_capturing_forward not found in {adapter_path}"
    )


def extract_upstream_forward_source() -> str:
    """Return the source of upstream MistralAttention.forward."""
    from transformers.models.mistral.modeling_mistral import MistralAttention

    return inspect.getsource(MistralAttention.forward)


def normalise(body: str) -> list[str]:
    """Normalise a forward body for comparison.

    Transforms applied in order:
      * Remove SAD INSERTION N .. END INSERTION N blocks
      * Drop decorator lines (e.g. @deprecate_kwarg) — upstream only, safe both sides
      * Drop ``# type: ignore[...]`` suffix comments (patched side only)
      * Convert ``Optional[X]`` -> ``X | None`` (upstream typing.Optional vs PEP 604)
      * Drop ``Unpack[FlashAttentionKwargs]`` annotation on ``**kwargs``
      * Substitute ``self`` -> ``module`` on both sides (closure capture)
      * Strip leading and trailing whitespace from every line
      * Drop blank lines

    After normalisation, remaining diffs are either (a) genuine drift, or
    (b) one of the known deliberate departures documented in SKILL.md.
    """
    stripped = INSERTION_BLOCK_RE.sub("", body)
    stripped = DECORATOR_RE.sub("", stripped)
    stripped = TYPE_IGNORE_RE.sub("", stripped)
    stripped = OPTIONAL_RE.sub(lambda m: f"{m.group(1)} | None", stripped)
    stripped = stripped.replace("**kwargs: Unpack[FlashAttentionKwargs]", "**kwargs")
    stripped = SELF_RE.sub("module", stripped)
    # Strip all leading/trailing whitespace per line. The Python parser already
    # enforces structural correctness, so any real indentation change will also
    # change the logical content (a missing if/else header line will show).
    # Dropping whitespace here focuses the diff on logical changes.
    return [line.strip() for line in stripped.splitlines() if line.strip()]


def main() -> int:
    try:
        patched_source = extract_patched_forward_source(ADAPTER_PATH)
        upstream_source = extract_upstream_forward_source()
    except Exception as exc:
        print(f"[adapter-upstream-diff] extraction failed: {exc}", file=sys.stderr)
        return 2

    patched_lines = normalise(patched_source)
    upstream_lines = normalise(upstream_source)

    diff = list(
        difflib.unified_diff(
            upstream_lines,
            patched_lines,
            fromfile="transformers upstream MistralAttention.forward (normalised)",
            tofile="navi-SAD patched_forward (insertions stripped, self->module)",
            lineterm="",
        )
    )

    if not diff:
        print(
            "[adapter-upstream-diff] OK: normalised bodies are identical "
            "(insertions and known departures excluded)."
        )
        return 0

    print(
        "[adapter-upstream-diff] DIFF (scan each hunk against SKILL.md's "
        "Known deliberate departures; anything unmatched is drift):"
    )
    for line in diff:
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
