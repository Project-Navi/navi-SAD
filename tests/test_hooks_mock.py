"""Tests for mock hook manager: compute_sad_delta and HookManager plumbing."""

import torch
import torch.nn as nn

from navi_sad.core.hooks import HookManager, compute_sad_delta

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
HIDDEN = 64
NUM_HEADS = 4
HEAD_DIM = 16


def _randn(shape: tuple[int, ...], seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


class FakeAttention(nn.Module):
    """Minimal attention module for testing hook plumbing."""

    def __init__(
        self,
        hidden_size: int = HIDDEN,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn_out)


# ===========================================================================
# TestComputeSadDelta
# ===========================================================================
class TestComputeSadDelta:
    def test_output_shape(self) -> None:
        """q=[1,4,1,16], k=[1,4,10,16], v same -> output shape [4]."""
        q = _randn((1, 4, 1, 16))
        k = _randn((1, 4, 10, 16), seed=7)
        v = _randn((1, 4, 10, 16), seed=13)
        delta = compute_sad_delta(q, k, v)
        assert delta.shape == (4,)

    def test_delta_range(self) -> None:
        """Random tensors -> all deltas between 0 and 2."""
        q = _randn((1, 4, 1, 16))
        k = _randn((1, 4, 10, 16), seed=7)
        v = _randn((1, 4, 10, 16), seed=13)
        delta = compute_sad_delta(q, k, v)
        assert (delta >= 0.0).all(), f"Delta below 0: {delta}"
        assert (delta <= 2.0).all(), f"Delta above 2: {delta}"

    def test_sink_exclusion_changes_result(self) -> None:
        """Deltas with sink_exclude=0 vs sink_exclude=2 must differ."""
        q = _randn((1, 4, 1, 16))
        k = _randn((1, 4, 10, 16), seed=7)
        v = _randn((1, 4, 10, 16), seed=13)
        delta_no_sink = compute_sad_delta(q, k, v, sink_exclude=0)
        delta_with_sink = compute_sad_delta(q, k, v, sink_exclude=2)
        assert not torch.equal(delta_no_sink, delta_with_sink), (
            "Sink exclusion should change the result"
        )

    def test_sink_exclusion_short_prefix(self) -> None:
        """If prefix length <= sink_exclude, must not crash."""
        q = _randn((1, 4, 1, 16))
        k = _randn((1, 4, 3, 16), seed=7)
        v = _randn((1, 4, 3, 16), seed=13)
        # sink_exclude=3 means skip all 3 positions — should not crash
        delta = compute_sad_delta(q, k, v, sink_exclude=3)
        assert delta.shape == (4,)
        assert torch.isfinite(delta).all()


# ===========================================================================
# TestHookManager
# ===========================================================================
class TestHookManager:
    def _make_module(self) -> FakeAttention:
        torch.manual_seed(0)
        return FakeAttention()

    def test_install_uninstall(self) -> None:
        """Install hooks -> is_installed True. Uninstall -> is_installed False."""
        attn = self._make_module()
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)
        assert hm.is_installed

        # Forward should still work with hooks installed
        x = _randn((1, 5, HIDDEN))
        attn(x)

        hm.uninstall()
        assert not hm.is_installed

    def test_non_interference(self) -> None:
        """Output MUST be identical with and without hooks.

        This is the most important test. The observer must not perturb the system.
        """
        attn = self._make_module()
        x = _randn((1, 5, HIDDEN))

        # Baseline: forward without hooks
        with torch.no_grad():
            baseline = attn(x).clone()

        # Install hooks and forward again
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)
        with torch.no_grad():
            hooked = attn(x).clone()

        hm.uninstall()

        assert torch.equal(baseline, hooked), (
            f"Hook installation changed model output!\n"
            f"Max diff: {(baseline - hooked).abs().max().item()}"
        )

    def test_captures_records(self) -> None:
        """Install hooks, run forward with seq_len=5, get_records returns non-empty."""
        attn = self._make_module()
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)

        x = _randn((1, 5, HIDDEN))
        with torch.no_grad():
            attn(x)

        records = hm.get_records()
        assert len(records) > 0, "Expected at least one StepRecord after forward"

        hm.uninstall()

    def test_reset_clears(self) -> None:
        """Install, forward, reset -> get_records returns empty."""
        attn = self._make_module()
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)

        x = _randn((1, 5, HIDDEN))
        with torch.no_grad():
            attn(x)

        assert len(hm.get_records()) > 0
        hm.reset()
        assert len(hm.get_records()) == 0, "reset() should clear all records"

        hm.uninstall()

    def test_step_counting(self) -> None:
        """Two forwards with step() between -> records have step_idx 0 and 1."""
        attn = self._make_module()
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)

        x = _randn((1, 5, HIDDEN))
        with torch.no_grad():
            attn(x)  # step_idx=0
        hm.step()
        with torch.no_grad():
            attn(x)  # step_idx=1

        records = hm.get_records()
        step_indices = sorted({r.step_idx for r in records})
        assert step_indices == [0, 1], f"Expected step indices [0, 1], got {step_indices}"

        hm.uninstall()

    def test_capture_cleanup(self) -> None:
        """After forward, _sad_capture should NOT exist on the module."""
        attn = self._make_module()
        hm = HookManager(sink_exclude=1)
        hm.install_on_module(attn, layer_idx=0, num_q_heads=NUM_HEADS, num_kv_heads=NUM_HEADS)

        x = _randn((1, 5, HIDDEN))
        with torch.no_grad():
            attn(x)

        assert not hasattr(attn, "_sad_capture"), (
            "_sad_capture must be deleted after post-hook processes it"
        )

        hm.uninstall()
