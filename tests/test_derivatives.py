"""Tests for finite differences on per-token delta series."""

import math

from navi_sad.signal.derivatives import compute_derivatives


# ===========================================================================
# TestComputeDerivatives
# ===========================================================================
class TestComputeDerivatives:
    def test_constant_zero_derivatives(self) -> None:
        """Constant sequence [0.5]*10 should produce derivatives all near zero."""
        result = compute_derivatives([0.5] * 10)
        for v in result["delta_prime"]:
            assert math.isclose(v, 0.0, abs_tol=1e-12), f"Expected ~0, got {v}"
        for v in result["delta_double_prime"]:
            assert math.isclose(v, 0.0, abs_tol=1e-12), f"Expected ~0, got {v}"
        for v in result["delta_triple_prime"]:
            assert math.isclose(v, 0.0, abs_tol=1e-12), f"Expected ~0, got {v}"

    def test_linear_constant_first_derivative(self) -> None:
        """Linear sequence [0.1*i] should have delta_prime all approx 0.1."""
        result = compute_derivatives([0.1 * i for i in range(10)])
        for v in result["delta_prime"]:
            assert math.isclose(v, 0.1, rel_tol=1e-9), f"Expected ~0.1, got {v}"

    def test_lengths(self) -> None:
        """20-element input: prime has 19, double_prime 18, triple_prime 17."""
        result = compute_derivatives(list(range(20)))
        assert len(result["delta_prime"]) == 19
        assert len(result["delta_double_prime"]) == 18
        assert len(result["delta_triple_prime"]) == 17

    def test_empty(self) -> None:
        """Empty input should produce all empty lists."""
        result = compute_derivatives([])
        assert result["delta_prime"] == []
        assert result["delta_double_prime"] == []
        assert result["delta_triple_prime"] == []

    def test_short_single_element(self) -> None:
        """Single element [0.5] should produce all empty derivative lists."""
        result = compute_derivatives([0.5])
        assert result["delta_prime"] == []
        assert result["delta_double_prime"] == []
        assert result["delta_triple_prime"] == []

    def test_short_two_elements(self) -> None:
        """Two elements [0.5, 0.6] should produce prime with 1 value, others empty."""
        result = compute_derivatives([0.5, 0.6])
        assert len(result["delta_prime"]) == 1
        assert math.isclose(result["delta_prime"][0], 0.1, rel_tol=1e-9)
        assert result["delta_double_prime"] == []
        assert result["delta_triple_prime"] == []
