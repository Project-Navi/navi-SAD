"""Effect size computation with validity guards.

Pure statistical functions. No domain-specific logic.
"""

from __future__ import annotations

import math

# Type alias for guarded statistics (value, reason).
# (float, None) when valid. (None, "reason string") when invalid.
GuardedStat = tuple[float | None, str | None]


def compute_cohens_d(
    group_a: list[float],
    group_b: list[float],
) -> GuardedStat:
    """Compute Cohen's d with validity guards.

    Returns (d, None) when valid, (None, reason) when invalid.
    Requires >= 2 samples per group and nonzero pooled variance.
    """
    if len(group_a) < 2:
        return (None, f"group_a has < 2 samples (n={len(group_a)})")
    if len(group_b) < 2:
        return (None, f"group_b has < 2 samples (n={len(group_b)})")

    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)

    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)

    n_a = len(group_a)
    n_b = len(group_b)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)

    if pooled_var == 0.0:
        return (None, "pooled variance is zero")

    d = (mean_a - mean_b) / math.sqrt(pooled_var)
    return (d, None)
