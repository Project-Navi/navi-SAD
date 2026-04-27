"""Shared fixtures for gate tests. Require GPU and real model.

`CALIBRATION_CAPABILITY` records the GPU compute capability that gate
tolerances were calibrated on. Gate runs assert this strictly via the
`_gate_environment_setup` autouse fixture; running on a different
compute capability requires re-calibration before the gates can be
trusted.
"""

from __future__ import annotations

import dataclasses

import pytest
import structlog
import torch

from navi_sad.core.environment import (
    assert_compatible_capability,
    capture_environment,
    enable_deterministic_mode,
)

log = structlog.get_logger()


# Compute capability the gate tolerances were calibrated on.
# (8, 6) = sm_86 = Ampere generation (RTX 3090, RTX A6000).
# Updating this constant is a frozen-decision change: gate tolerances
# would need to be re-calibrated and re-frozen on the new capability.
CALIBRATION_CAPABILITY: tuple[int, int] = (8, 6)


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Auto-skip @pytest.mark.gpu tests when no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def _gate_environment_setup():
    """Enable deterministic mode and assert calibration capability.

    Runs once per session before any gate test. On no-GPU hosts the
    capability assertion is skipped (the gpu-marker auto-skip handles
    test selection). On GPU hosts, a mismatched compute capability
    fails the session immediately rather than producing tolerance-pass
    results that are not trustworthy.
    """
    enable_deterministic_mode()
    if torch.cuda.is_available():
        assert_compatible_capability(CALIBRATION_CAPABILITY, strict=True)
        snapshot = capture_environment()
        log.info("gate_environment_captured", **dataclasses.asdict(snapshot))
    yield


@pytest.fixture(scope="session")
def mistral_model_and_tokenizer():
    """Load Mistral-7B-Instruct-v0.2 once per test session.

    Uses fp16 (frozen decision: fp16 only for verification gates).
    Uses eager attention (required for forward-replacement adapter).
    Determinism flags are set in `_gate_environment_setup`; this
    fixture only adds RNG seeds (local to model loading state, not
    global determinism config).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # Pin revision for reproducibility. Update only after re-validating gates.
    revision = "63a8b081895390a26e140280378bc85ec8bce07a"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer
