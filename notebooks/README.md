# notebooks/

Exploratory Jupyter notebooks. Not part of the test suite; not run in CI.

## Setup

The notebook tooling lives in the `notebook` extra in `pyproject.toml`
(`jupyterlab`, `ipykernel`, `ordpy`, `ipywidgets`, `seaborn`). The
demo notebook also imports `pandas` and `matplotlib`, which live in
the `analysis` extra (alongside `scipy` and `scikit-learn`). To
install both:

```bash
uv sync --extra notebook --extra analysis
```

`uv` does not install optional extras unless explicitly requested
([uv sync docs](https://docs.astral.sh/uv/concepts/projects/sync/)),
so a clean environment that only passes `--extra notebook` will
fail on `import pandas` / `import matplotlib`. Both extras are
required.

This adds the deps to `.venv` and registers a project-local kernel at
`.venv/share/jupyter/kernels/python3` — the kernelspec travels with the
venv, so the setup is reproducible per `uv.lock`.

## Launching JupyterLab

```bash
uv run --extra notebook --extra analysis jupyter lab --no-browser --ip=127.0.0.1
```

JupyterLab prints a URL with a session token (e.g.
`http://127.0.0.1:8888/lab?token=…`) — open that.

## Choosing a kernel

In the kernel picker (top-right of any open notebook), select:

**`Python 3 (ipykernel)`**

That kernel uses `.venv/bin/python` from this project, so all of
navi-SAD's locked deps (torch, transformers, numpy, scipy, ordpy,
ipywidgets, etc.) are importable. There is no separate user-level
kernel registration step — the kernel ships with the venv when you
`uv sync --extra notebook --extra analysis`.

If you see other kernels in the picker (e.g. from older user-level
`ipykernel install --user --name foo` calls), ignore them. The
project-local `Python 3 (ipykernel)` is the only one whose env
matches `pyproject.toml + uv.lock`.

## Notebooks here

| File | Purpose |
|------|---------|
| `renyi_entropy_complexity_transformer_interpretability_demo.ipynb` | Adapts Guisande & Montani (2024) Rényi entropy-complexity causality plane to transformer activations. Source / starting point for the navi-SAD-side Rényi fingerprint port. Uses `ordpy` for ordinal patterns. |

## House rules

- Notebooks are **exploratory**. Anything load-bearing (fingerprint code,
  statistical tests, gate logic) belongs in `src/navi_sad/` with tests,
  not in a notebook.
- The notebook output cells **may** be committed (they show the
  intended results) but `.ipynb_checkpoints/` is gitignored.
- If a notebook depends on a new package, add it to the `notebook`
  extra in `pyproject.toml` rather than relying on the notebook's
  `pip install` bootstrap cells. Locked deps > runtime installs.
- Notebooks should not write to `results/` (which is gitignored
  everywhere) without an explicit subdir under `results/notebooks/`
  to keep them separate from gate / pilot artifacts.
