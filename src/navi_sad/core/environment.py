"""Runtime environment capture and reproducibility helpers.

Three responsibilities:

1. **Capture**: produce an immutable `EnvironmentSnapshot` recording
   the torch / CUDA / cuDNN / GPU state at a given moment. Useful for
   logging at instrument startup and for attaching to persisted
   artifacts.
2. **Determinism**: `enable_deterministic_mode()` centralises the
   torch flags (cudnn.deterministic, cudnn.benchmark, the
   `CUBLAS_WORKSPACE_CONFIG` env var) that affect GPU numerical
   reproducibility. Idempotent.
3. **Capability assertion**: `assert_compatible_capability()` enforces
   that the running GPU's compute capability matches what the gates
   were calibrated for. Strict (raise) for gate runs; non-strict
   (warn) for casual research.

Note on what this module does NOT do:

- It does not pin the NVIDIA driver version (kernel-level).
- It does not pin the OS / glibc / kernel.
- It does not call `torch.use_deterministic_algorithms(True)` — that
  is more aggressive and breaks any non-deterministic op (some
  research code may rely on those). Add per-call-site if needed.
"""

from __future__ import annotations

import dataclasses
import os

import structlog
import torch

log = structlog.get_logger()


class CapabilityMismatchError(RuntimeError):
    """Raised by `assert_compatible_capability(strict=True)` when the
    running GPU's compute capability does not match the expected value."""


@dataclasses.dataclass(frozen=True)
class EnvironmentSnapshot:
    """Immutable record of torch / CUDA / GPU state.

    Fields default to `None` or `0` when CUDA is not available so the
    snapshot remains well-formed on CPU-only hosts.
    """

    torch_version: str
    cuda_compile_version: str | None
    cudnn_version: int | None
    gpu_name: str | None
    gpu_capability: tuple[int, int] | None
    gpu_count: int
    deterministic_mode: bool


def capture_environment() -> EnvironmentSnapshot:
    """Snapshot the current torch / CUDA environment.

    Pure observation — does not modify state.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name: str | None = torch.cuda.get_device_name(0) if gpu_count else None
        gpu_capability: tuple[int, int] | None = (
            torch.cuda.get_device_capability(0) if gpu_count else None
        )
        cudnn_version: int | None = torch.backends.cudnn.version()  # type: ignore[attr-defined]
        cuda_compile_version: str | None = torch.version.cuda  # type: ignore[attr-defined]
    else:
        gpu_count = 0
        gpu_name = None
        gpu_capability = None
        cudnn_version = None
        cuda_compile_version = None

    return EnvironmentSnapshot(
        torch_version=torch.__version__,
        cuda_compile_version=cuda_compile_version,
        cudnn_version=cudnn_version,
        gpu_name=gpu_name,
        gpu_capability=gpu_capability,
        gpu_count=gpu_count,
        deterministic_mode=bool(torch.backends.cudnn.deterministic),  # type: ignore[attr-defined]
    )


def enable_deterministic_mode() -> None:
    """Set torch + cuBLAS flags for GPU numerical reproducibility.

    Specifically:
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`
    - `CUBLAS_WORKSPACE_CONFIG=":16:8"` (only if not already set to a
      deterministic value)

    Idempotent. Affects global state. Does not call
    `torch.use_deterministic_algorithms(True)` — see module docstring.
    """
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    # PyTorch documents two acceptable values for deterministic cuBLAS GEMM:
    # ":16:8" (smaller workspace) and ":4096:8" (more workspace, same
    # determinism guarantee). If the user has already set either, do
    # not overwrite. Otherwise set ":16:8".
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (":16:8", ":4096:8"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def assert_compatible_capability(
    expected: tuple[int, int],
    *,
    strict: bool = True,
) -> None:
    """Assert the running GPU's compute capability matches `expected`.

    Args:
        expected: `(major, minor)` tuple, e.g. `(8, 6)` for sm_86
            (Ampere / RTX 3090).
        strict: If `True`, raise `CapabilityMismatchError` on mismatch.
            If `False`, log a warning and continue. Either way, raises
            `RuntimeError` if no CUDA device is available.

    Raises:
        RuntimeError: No CUDA device available.
        CapabilityMismatchError: `strict=True` and capability differs.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available; capability assertion requires GPU. "
            "If running CPU-only tests, do not call this function."
        )
    actual = torch.cuda.get_device_capability(0)
    if actual == expected:
        return
    msg = (
        f"GPU compute capability mismatch: expected {expected} "
        f"(sm_{expected[0]}{expected[1]}), found {actual} "
        f"(sm_{actual[0]}{actual[1]}). Gate-parity tolerances were "
        f"calibrated on the expected capability; results on different "
        f"capabilities may differ enough to require re-calibration."
    )
    if strict:
        raise CapabilityMismatchError(msg)
    log.warning(
        "capability_mismatch",
        expected=expected,
        actual=actual,
        message=msg,
    )
