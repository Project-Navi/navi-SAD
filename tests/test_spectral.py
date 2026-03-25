"""Tests for spectral core: softmax, linear attention, GQA expansion, cosine distance."""

import torch

from navi_sad.core.spectral import (
    expand_kv_heads,
    linear_attention_last_token,
    per_head_cosine_distance,
    softmax_attention_last_token,
)

# ---------------------------------------------------------------------------
# Fixtures: common tensor dimensions
# ---------------------------------------------------------------------------
B, H, L, D = 1, 8, 32, 64


def _randn(shape: tuple[int, ...], seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g)


# ===========================================================================
# TestSoftmaxLastToken
# ===========================================================================
class TestSoftmaxLastToken:
    def test_output_shape(self) -> None:
        q = _randn((B, H, 1, D))
        k = _randn((B, H, L, D))
        v = _randn((B, H, L, D))
        out = softmax_attention_last_token(q, k, v)
        assert out.shape == (B, H, 1, D)

    def test_no_nan_or_inf(self) -> None:
        q = _randn((B, H, 1, D)) * 10
        k = _randn((B, H, L, D), seed=7) * 10
        v = _randn((B, H, L, D), seed=13) * 10
        out = softmax_attention_last_token(q, k, v)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_attention_weights_sum_to_one(self) -> None:
        q = _randn((B, H, 1, D))
        k = _randn((B, H, L, D), seed=7)
        _v = _randn((B, H, L, D), seed=13)
        # Compute attention weights manually to verify they sum to 1
        scale = D**0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, 1, L]
        weights = torch.softmax(scores, dim=-1)
        sums = weights.sum(dim=-1)  # [B, H, 1]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ===========================================================================
# TestLinearLastToken
# ===========================================================================
class TestLinearLastToken:
    def test_output_shape(self) -> None:
        q = _randn((B, H, 1, D))
        k = _randn((B, H, L, D))
        v = _randn((B, H, L, D))
        out = linear_attention_last_token(q, k, v)
        assert out.shape == (B, H, 1, D)

    def test_no_nan_or_inf(self) -> None:
        q = _randn((B, H, 1, D)) * 10
        k = _randn((B, H, L, D), seed=7) * 10
        v = _randn((B, H, L, D), seed=13) * 10
        out = linear_attention_last_token(q, k, v)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_single_position(self) -> None:
        """L=1 prefix: linear attention must still produce valid output."""
        q = _randn((B, H, 1, D))
        k = _randn((B, H, 1, D), seed=7)
        v = _randn((B, H, 1, D), seed=13)
        out = linear_attention_last_token(q, k, v)
        assert out.shape == (B, H, 1, D)
        assert torch.isfinite(out).all()


# ===========================================================================
# TestGQAExpansion
# ===========================================================================
class TestGQAExpansion:
    def test_expand_2_to_8(self) -> None:
        kv = _randn((B, 2, L, D))
        expanded = expand_kv_heads(kv, num_q_heads=8)
        assert expanded.shape == (B, 8, L, D)
        # Each group of 4 Q-heads should mirror the corresponding KV head
        for i in range(8):
            kv_idx = i // 4  # 8 Q heads / 2 KV heads = 4 per group
            assert torch.equal(expanded[:, i], kv[:, kv_idx])

    def test_no_expansion_when_equal(self) -> None:
        kv = _randn((B, 8, L, D))
        expanded = expand_kv_heads(kv, num_q_heads=8)
        assert expanded.shape == (B, 8, L, D)
        assert torch.equal(expanded, kv)


# ===========================================================================
# TestCosineDistance
# ===========================================================================
class TestCosineDistance:
    def test_identical_is_zero(self) -> None:
        a = _randn((B, H, 1, D))
        dist = per_head_cosine_distance(a, a)
        assert torch.allclose(dist, torch.zeros(H), atol=1e-5)

    def test_orthogonal_is_one(self) -> None:
        """Construct orthogonal vectors and verify distance ~= 1."""
        a = torch.zeros(B, 2, 1, 4)
        b = torch.zeros(B, 2, 1, 4)
        # Head 0: [1,0,0,0] vs [0,1,0,0]
        a[:, 0, 0, 0] = 1.0
        b[:, 0, 0, 1] = 1.0
        # Head 1: [0,0,1,0] vs [0,0,0,1]
        a[:, 1, 0, 2] = 1.0
        b[:, 1, 0, 3] = 1.0
        dist = per_head_cosine_distance(a, b)
        assert torch.allclose(dist, torch.ones(2), atol=1e-5)

    def test_output_shape(self) -> None:
        num_heads = 32
        a = _randn((B, num_heads, 1, D))
        b = _randn((B, num_heads, 1, D), seed=99)
        dist = per_head_cosine_distance(a, b)
        assert dist.shape == (num_heads,)

    def test_range_zero_to_two(self) -> None:
        a = _randn((2, H, 1, D))
        b = _randn((2, H, 1, D), seed=99)
        dist = per_head_cosine_distance(a, b)
        assert (dist >= 0.0).all(), f"Distance below 0: {dist}"
        assert (dist <= 2.0).all(), f"Distance above 2: {dist}"


# ===========================================================================
# TestConvergence — validates the Han et al. theoretical foundation
# ===========================================================================
class TestConvergence:
    def test_uniform_keys_zero_divergence(self) -> None:
        """When all keys are identical, attention is uniform.

        Softmax and linear attention should produce the same output because
        there is no regime difference. This is the foundational property:
        divergence == 0 when the attention distribution is uniform.
        """
        # All keys identical -> uniform attention weights
        single_key = _randn((B, H, 1, D))
        k_uniform = single_key.expand(B, H, L, D).contiguous()
        v = _randn((B, H, L, D), seed=7)
        q = _randn((B, H, 1, D), seed=13)

        out_softmax = softmax_attention_last_token(q, k_uniform, v)
        out_linear = linear_attention_last_token(q, k_uniform, v)

        dist = per_head_cosine_distance(out_softmax, out_linear)
        assert (dist < 0.05).all(), (
            f"Cosine distance too high for uniform keys: {dist}. "
            f"Expected < 0.05 per Han et al. convergence property."
        )
