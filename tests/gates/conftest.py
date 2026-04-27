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
# (8, 6) = sm_86 = Ampere generation (RTX 3090/3080/3070/A6000/A5000/...).
#
# Capability is a *necessary* condition: SASS instructions and PTX ISA
# are tied to capability, so any compatibility break (e.g., sm_80 vs
# sm_86) invalidates the gate tolerances. Capability is *not* a
# sufficient condition: different sm_86 GPUs can produce slightly
# different fp16 numerics due to driver version, SM count, memory
# bandwidth, and accumulation order differences. The autouse fixture
# strict-asserts capability and warns on GPU-name mismatch so gate-pass
# trust degrades gracefully instead of silently. Updating either
# constant is a frozen-decision change.
CALIBRATION_CAPABILITY: tuple[int, int] = (8, 6)

# Specific GPU where Gate 1 tolerances were calibrated (2026-03-24).
# Running on a different sm_86 GPU is plausible but not verified to pass
# tolerances; the fixture warns rather than fails so this can surface.
# Driver version at calibration time was not recorded; the snapshot log
# captures the current driver/CUDA versions for audit.
CALIBRATION_GPU_NAME: str = "NVIDIA GeForce RTX 3090"


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Auto-skip @pytest.mark.gpu tests when no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session", autouse=True)
def _gate_environment_setup():
    """Enable deterministic mode and check calibration hardware.

    Runs once per session before any gate test. On no-GPU hosts the
    capability check is skipped (the gpu-marker auto-skip handles test
    selection).

    On GPU hosts:
    - Strict-asserts `CALIBRATION_CAPABILITY` (necessary condition for
      SASS compatibility — a mismatch would silently invalidate the
      frozen gate tolerances).
    - Soft-warns on `CALIBRATION_GPU_NAME` mismatch (sufficient
      condition for trustworthy tolerance pass — can plausibly hold on
      other sm_86 GPUs but is not verified, so we surface the
      divergence rather than fail).
    - Captures and logs the full environment snapshot at INFO so any
      driver/CUDA/cuDNN drift since the original calibration is on
      record in the run output.
    """
    enable_deterministic_mode()
    if torch.cuda.is_available():
        assert_compatible_capability(CALIBRATION_CAPABILITY, strict=True)
        snapshot = capture_environment()
        log.info("gate_environment_captured", **dataclasses.asdict(snapshot))
        if snapshot.gpu_name != CALIBRATION_GPU_NAME:
            log.warning(
                "calibration_gpu_mismatch",
                expected=CALIBRATION_GPU_NAME,
                actual=snapshot.gpu_name,
                message=(
                    "Gate tolerances were calibrated on a specific GPU "
                    "model. Compute capability matches (necessary "
                    "condition) but the running GPU differs. Tolerance "
                    "pass is plausible but not verified for this "
                    "hardware; treat results accordingly."
                ),
            )
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
