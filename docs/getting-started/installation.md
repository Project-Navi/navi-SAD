# Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

## Quick install

```bash
git clone https://github.com/Project-Navi/navi-SAD.git
cd navi-SAD
uv sync --extra dev
```

## Optional dependency groups

| Group | What it adds | Install command |
|-------|-------------|----------------|
| `dev` | pytest, ruff, mypy, pre-commit | `uv sync --extra dev` |
| `eval` | datasets (HuggingFace) | `uv sync --extra dev --extra eval` |
| `analysis` | scipy, scikit-learn, matplotlib, pandas | `uv sync --extra dev --extra analysis` |

## Verify

```bash
make all
```

This runs lint + format check + typecheck + CPU tests (no GPU required).
