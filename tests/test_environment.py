"""Tests for navi_sad.core.environment — runtime hardware/CUDA capture
and reproducibility-related setup helpers.

Most tests are CPU-only by mocking torch.cuda. The end-to-end snapshot
on real CUDA is GPU-marked separately."""

from __future__ import annotations

import dataclasses
import os
from unittest.mock import patch

import pytest
import torch

from navi_sad.core.environment import (
    CapabilityMismatchError,
    EnvironmentSnapshot,
    assert_compatible_capability,
    capture_environment,
    enable_deterministic_mode,
)


@pytest.fixture
def restore_torch_backends():
    """Save and restore torch.backends.cudnn flags + env var across tests."""
    original_deterministic = torch.backends.cudnn.deterministic
    original_benchmark = torch.backends.cudnn.benchmark
    original_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.backends.cudnn.deterministic = original_deterministic  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = original_benchmark  # type: ignore[attr-defined]
    if original_workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_workspace


class TestEnvironmentSnapshot:
    def test_snapshot_is_frozen(self):
        snap = EnvironmentSnapshot(
            torch_version="2.11.0+cu130",
            cuda_compile_version="13.0",
            cudnn_version=91900,
            gpu_name="NVIDIA GeForce RTX 3090",
            gpu_capability=(8, 6),
            gpu_count=1,
            deterministic_mode=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.torch_version = "other"  # type: ignore[misc]

    def test_snapshot_supports_asdict(self):
        snap = EnvironmentSnapshot(
            torch_version="2.11.0+cu130",
            cuda_compile_version=None,
            cudnn_version=None,
            gpu_name=None,
            gpu_capability=None,
            gpu_count=0,
            deterministic_mode=False,
        )
        d = dataclasses.asdict(snap)
        assert d["torch_version"] == "2.11.0+cu130"
        assert d["gpu_count"] == 0
        assert d["gpu_capability"] is None


class TestCaptureEnvironment:
    def test_capture_no_cuda_returns_none_for_gpu_fields(self):
        with patch("torch.cuda.is_available", return_value=False):
            snap = capture_environment()
        assert snap.gpu_count == 0
        assert snap.gpu_name is None
        assert snap.gpu_capability is None
        assert snap.cudnn_version is None
        # torch_version is always available
        assert snap.torch_version == torch.__version__

    def test_capture_records_deterministic_state(self, restore_torch_backends):
        with patch("torch.cuda.is_available", return_value=False):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            snap = capture_environment()
            assert snap.deterministic_mode is True

            torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
            snap2 = capture_environment()
            assert snap2.deterministic_mode is False

    def test_capture_returns_immutable_snapshot(self):
        with patch("torch.cuda.is_available", return_value=False):
            snap = capture_environment()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.gpu_count = 99  # type: ignore[misc]


class TestEnableDeterministicMode:
    def test_sets_cudnn_deterministic(self, restore_torch_backends):
        torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
        enable_deterministic_mode()
        assert torch.backends.cudnn.deterministic is True  # type: ignore[attr-defined]

    def test_disables_cudnn_benchmark(self, restore_torch_backends):
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        enable_deterministic_mode()
        assert torch.backends.cudnn.benchmark is False  # type: ignore[attr-defined]

    def test_sets_cublas_workspace_config(self, restore_torch_backends):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        enable_deterministic_mode()
        # PyTorch accepts ":16:8" or ":4096:8" for deterministic cuBLAS GEMM
        # We use ":16:8" (smaller workspace, deterministic).
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"

    def test_idempotent(self, restore_torch_backends):
        enable_deterministic_mode()
        enable_deterministic_mode()
        enable_deterministic_mode()
        assert torch.backends.cudnn.deterministic is True  # type: ignore[attr-defined]
        assert torch.backends.cudnn.benchmark is False  # type: ignore[attr-defined]
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":16:8"

    def test_does_not_clobber_existing_workspace_config(self, restore_torch_backends):
        # If the user (or environment) has already set a stricter config,
        # we should not weaken it. ":4096:8" is also deterministic and
        # uses more workspace; we accept it.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        enable_deterministic_mode()
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


class TestAssertCompatibleCapability:
    def test_raises_runtime_error_when_no_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="No CUDA"):
                assert_compatible_capability((8, 6))

    def test_passes_on_match_strict(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            # Should not raise.
            assert_compatible_capability((8, 6), strict=True)

    def test_raises_capability_mismatch_strict(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            with pytest.raises(CapabilityMismatchError, match="9, 0"):
                assert_compatible_capability((8, 6), strict=True)

    def test_warns_on_mismatch_non_strict(self):
        import structlog

        with (
            structlog.testing.capture_logs() as captured,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
        ):
            assert_compatible_capability((8, 6), strict=False)
        events = [c for c in captured if c.get("event") == "capability_mismatch"]
        assert len(events) == 1
        assert events[0]["log_level"] == "warning"
        assert events[0]["expected"] == (8, 6)
        assert events[0]["actual"] == (8, 0)

    def test_passes_on_match_non_strict(self):
        import structlog

        with (
            structlog.testing.capture_logs() as captured,
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        ):
            assert_compatible_capability((8, 6), strict=False)
        # No capability_mismatch event should be emitted on match.
        events = [c for c in captured if c.get("event") == "capability_mismatch"]
        assert events == []


@pytest.mark.gpu
class TestCaptureEnvironmentOnGpu:
    """End-to-end snapshot on real CUDA. Requires GPU."""

    def test_gpu_fields_populated_on_real_cuda(self):
        snap = capture_environment()
        assert snap.gpu_count >= 1
        assert snap.gpu_name is not None
        assert snap.gpu_name != ""
        assert snap.gpu_capability is not None
        assert isinstance(snap.gpu_capability, tuple)
        assert len(snap.gpu_capability) == 2
        assert snap.cuda_compile_version is not None
        assert snap.cudnn_version is not None
        assert snap.cudnn_version > 0
