# Claude Code Hooks — navi-SAD

Two project-scoped hooks are recommended for this repo but are **not checked in**, because the `.claude/settings.json` file that hosts them also typically holds personal permissions that should not be shared. If you want the behaviour, copy the snippets below into your own `.claude/settings.json` (or `.claude/settings.local.json` if you prefer to keep personal permissions in the canonical project file).

Both hooks are read from stdin, so they work uniformly across Claude Code versions that pass the tool-use payload as JSON on stdin.

---

## Hook 1 — Warn on `core/adapter.py` edits

**Why.** CLAUDE.md states: *"Any change requires Gate 0 re-verification."* Nothing currently enforces this — a routine Edit on `src/navi_sad/core/adapter.py` can silently break Gate 0 with no compile-time signal. This `PreToolUse` hook prints a loud warning before every adapter edit and reminds you to run `gate-check` plus `adapter-upstream-diff`.

**Snippet** — merge into your `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c 'import sys, json; d=json.load(sys.stdin); p=(d.get(\"tool_input\",{}) or {}).get(\"file_path\",\"\") or \"\"; sys.exit(0) if \"core/adapter.py\" not in p else (print(\"[navi-SAD] WARNING: core/adapter.py is a verbatim upstream copy. Any change requires Gate 0 + Gate 1 re-verification. Run /adapter-upstream-diff and /gate-check before merging.\", file=sys.stderr) or sys.exit(0))'"
          }
        ]
      }
    ]
  }
}
```

- Exits 0 (allows the edit) and writes the warning to stderr.
- To make the hook **blocking** (require an explicit acknowledgement ritual), replace the final `sys.exit(0)` with `sys.exit(2)` — Claude Code treats exit 2 as a hook veto.

---

## Hook 2 — ruff on Python edits

**Why.** Pre-commit runs ruff + mypy, but only at commit time. Running `ruff check --fix` and `ruff format` on every `.py` edit catches issues before you stack further edits on top, and matches the pinned tool versions without adding any new dependency.

**Snippet:**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 -c 'import sys, json, subprocess; d=json.load(sys.stdin); p=(d.get(\"tool_input\",{}) or {}).get(\"file_path\",\"\") or \"\"; ok = p.endswith(\".py\") and \"navi-SAD\" in p; sys.exit(0) if not ok else (subprocess.run([\"uv\",\"run\",\"--quiet\",\"ruff\",\"check\",\"--fix\",p], check=False), subprocess.run([\"uv\",\"run\",\"--quiet\",\"ruff\",\"format\",p], check=False), sys.exit(0))'"
          }
        ]
      }
    ]
  }
}
```

- Only runs on `.py` files whose path contains `navi-SAD` — prevents accidental runs when editing files in sibling repos during the same session.
- Errors from ruff are suppressed (`check=False`) so a genuine lint violation doesn't block the edit. Pre-commit still catches violations at commit time.

---

## Merging with existing hooks

If you already have hooks in `.claude/settings.json`, the JSON arrays under `PreToolUse` / `PostToolUse` merge by concatenation — just append these entries to the existing arrays rather than replacing them.

## Testing the hooks

After editing `.claude/settings.json`:

- **Hook 1**: trigger any `Edit` or `Write` on `src/navi_sad/core/adapter.py`. The warning should print to stderr before the edit is applied.
- **Hook 2**: trigger any `Edit` or `Write` on a `.py` file under this repo. After the edit, the file should be re-formatted (if it was out of format) — compare with `git diff`.

If a hook silently does nothing, run Claude Code with `--debug` or check `~/.claude/logs/` for hook stderr / exit codes.

## Why not commit `.claude/settings.json`?

The canonical `.claude/settings.json` also typically contains `permissions.allow` entries with user-specific paths (e.g., paths to sibling repos on the operator's machine, granted MCP tool names). Committing it would leak personal paths and force teammates to adopt a permission set that does not match their layout. Documenting the hooks here and letting each teammate paste them into their local settings is safer and keeps the shared repo free of per-user noise.

`.claude/settings.local.json` is already covered by the user's global git ignore and should never end up tracked.
