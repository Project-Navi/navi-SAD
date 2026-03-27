# Contributing

navi-SAD is a research project under active development. Contributions are welcome --- especially on the [open problems](../research/open-problems.md).

## Getting started

```bash
git clone https://github.com/Project-Navi/navi-SAD.git
cd navi-SAD
uv sync --extra dev
uv run pre-commit install
make all
```

## Workflow

1. Fork and create a feature branch (`feat/your-feature` or `fix/your-fix`)
2. Write tests first (TDD --- see the project's testing standards)
3. Run `make all` before committing (lint + format + typecheck + test)
4. Open a PR against `main`

## Commit conventions

`<type>: <description>` --- types: feat, fix, refactor, test, chore, docs

## Code of conduct

Be collegial. Critique ideas, not people. Negative results are information, not failure.
