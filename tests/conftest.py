"""Top-level pytest configuration shared across the whole test tree.

Two responsibilities here that need to fire BEFORE collection:
- Register the `gpu` marker.
- Apply `enable_deterministic_mode()` so `CUBLAS_WORKSPACE_CONFIG` and
  the cudnn flags are set before any plugin or test module probes
  CUDA. The pytest collection phase calls `torch.cuda.is_available()`
  in our `pytest_collection_modifyitems` hook below, and we want
  determinism config to land first.

Plus:
- Auto-skip `@pytest.mark.gpu` tests on hosts without CUDA. This used
  to live in `tests/gates/conftest.py`, but a few GPU-marked tests
  (e.g. `tests/test_environment.py`) live outside `tests/gates/` and
  weren't covered by the local hook.
"""

from __future__ import annotations

import pytest
import torch

from navi_sad.core.environment import enable_deterministic_mode


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")
    # Earliest pytest hook. Sets cudnn flags and CUBLAS_WORKSPACE_CONFIG
    # before any module-level imports, plugin code, or our own
    # `pytest_collection_modifyitems` hook can touch CUDA.
    enable_deterministic_mode()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.gpu tests when no GPU is available.

    Runs at collection time, after `pytest_configure` so determinism
    config is already in place.
    """
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
