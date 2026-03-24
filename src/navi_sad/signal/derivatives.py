"""Finite differences on the per-token delta series."""

import numpy as np


def compute_derivatives(delta: list[float]) -> dict:
    """Compute first, second, and third finite differences.

    Token-index spacing (dt=1). No normalization.

    Args:
        delta: per-token aggregated delta values

    Returns:
        dict with keys delta_prime, delta_double_prime, delta_triple_prime
        (each a list of float)
    """
    arr = np.array(delta, dtype=np.float64)
    return {
        "delta_prime": np.diff(arr, n=1).tolist() if len(arr) > 1 else [],
        "delta_double_prime": np.diff(arr, n=2).tolist() if len(arr) > 2 else [],
        "delta_triple_prime": np.diff(arr, n=3).tolist() if len(arr) > 3 else [],
    }
