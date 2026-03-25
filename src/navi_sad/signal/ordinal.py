"""Ordinal pattern extraction and permutation entropy.

Implements the Bandt-Pompe method for converting a scalar sequence into
ordinal (permutation) patterns, plus normalized Shannon permutation entropy.

Ported from the C++ kernel at:
    project-navi-api/kernel/include/navi_dsc_renyi.h  (lines 115-251)

Critical deviation from the C++ reference: windows containing ties
(any two values within *epsilon* of each other) are EXCLUDED from the
pattern list.  This prevents fake ordinal structure from inflating PE
where the theory predicts collapse.

References:
    Bandt, C. & Pompe, B. (2002). Permutation Entropy: A Natural
    Complexity Measure for Time Series.  Physical Review Letters 88(17).
"""

from __future__ import annotations

import math
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACTORIAL: list[int] = [1, 1, 2, 6, 24, 120, 720, 5040, 40320]


# ---------------------------------------------------------------------------
# PE eligibility policy
# ---------------------------------------------------------------------------
def recommended_min_pe_length(
    D: int,
    tau: int,
    min_windows: int | None = None,
) -> int:
    """Minimum sequence length for PE computation under the current policy.

    This is a **policy threshold**, not a mathematical requirement for PE
    to be defined.  PE is structurally computable with as few as one
    strict-order window, but statistically unreliable with too few.

    The default policy requires at least D! ordinal windows, which is the
    minimum for all D! possible patterns to appear at least once.

    Args:
        D: Embedding dimension.
        tau: Embedding delay.
        min_windows: Minimum number of ordinal windows required.
            Defaults to ``math.factorial(D)`` when ``None``.

    Returns:
        Minimum sequence length that yields at least *min_windows* windows.
    """
    if min_windows is None:
        min_windows = math.factorial(D)
    # n_windows = n - (D - 1) * tau  =>  n = min_windows + (D - 1) * tau
    return min_windows + (D - 1) * tau


# ---------------------------------------------------------------------------
# Lehmer code conversion
# ---------------------------------------------------------------------------
def permutation_to_index(perm: list[int], D: int) -> int:
    """Convert a permutation to its Lehmer-code index.

    Identical algorithm to the C++ ``permutation_to_index()``:
    for each position *i*, count how many later elements are smaller,
    multiply by the appropriate factorial, and accumulate.

    Parameters
    ----------
    perm : list[int]
        A permutation of ``range(D)`` expressed as rank assignments.
    D : int
        Embedding dimension (length of *perm*).

    Returns
    -------
    int
        Index in ``[0, D!)`` uniquely identifying the permutation.
    """
    index = 0
    factorial = FACTORIAL[D - 1]

    for i in range(D - 1):
        count = 0
        for j in range(i + 1, D):
            if perm[j] < perm[i]:
                count += 1
        index += count * factorial
        if D - 1 - i > 0:
            factorial //= D - 1 - i

    return index


# ---------------------------------------------------------------------------
# Ordinal pattern extraction (with tie exclusion)
# ---------------------------------------------------------------------------
def extract_ordinal_patterns(
    sequence: list[float] | list[int],
    D: int = 3,
    tau: int = 1,
    epsilon: float = 1e-9,
) -> tuple[list[int], int, float]:
    """Extract ordinal patterns from *sequence* using the Bandt-Pompe method.

    Windows where any two values are within *epsilon* of each other are
    counted as **tied** and excluded from the pattern list.

    Parameters
    ----------
    sequence : list[float] | list[int]
        Input scalar sequence.
    D : int
        Embedding dimension (number of elements per window).
    tau : int
        Embedding delay (step between elements within a window).
    epsilon : float
        Absolute tolerance for tie detection.

    Returns
    -------
    tuple[list[int], int, float]
        ``(strict_patterns, tied_count, tie_rate)`` where
        *strict_patterns* is a list of Lehmer-code indices,
        *tied_count* is the number of windows excluded due to ties, and
        *tie_rate* = tied_count / total_windows (0.0 when there are no
        windows).
    """
    n = len(sequence)
    n_patterns = n - (D - 1) * tau
    if n_patterns <= 0:
        return [], 0, 0.0

    strict_patterns: list[int] = []
    tied_count = 0

    for i in range(n_patterns):
        # Extract D values with delay tau
        window: list[float] = [sequence[i + j * tau] for j in range(D)]

        # --- tie detection: check all pairs ---
        has_tie = False
        for a in range(D):
            if has_tie:
                break
            for b in range(a + 1, D):
                if abs(window[a] - window[b]) <= epsilon:
                    has_tie = True
                    break

        if has_tie:
            tied_count += 1
            continue

        # --- compute ordinal pattern ---
        # Build (value, original_position) pairs, sort by value.
        # Position is the tiebreaker for deterministic ordering (stable sort
        # on position is implicit because we enumerate in order and Python's
        # sort is stable).
        indexed: list[tuple[float, int]] = [(window[j], j) for j in range(D)]
        indexed.sort(key=lambda x: x[0])

        # Assign ranks: the element that was at position indexed[j].second
        # receives rank j.
        permutation = [0] * D
        for rank, (_, original_pos) in enumerate(indexed):
            permutation[original_pos] = rank

        strict_patterns.append(permutation_to_index(permutation, D))

    total_windows = n_patterns
    tie_rate = tied_count / total_windows if total_windows > 0 else 0.0

    return strict_patterns, tied_count, tie_rate


# ---------------------------------------------------------------------------
# Permutation entropy (normalized Shannon)
# ---------------------------------------------------------------------------
def permutation_entropy(
    sequence: list[float] | list[int],
    D: int = 3,
    tau: int = 1,
    epsilon: float = 1e-9,
) -> tuple[float | None, float, dict[int, int]]:
    """Compute the normalized Shannon permutation entropy of *sequence*.

    PE is computed only over strict-order windows (ties excluded).
    Normalized to [0, 1] by dividing by ``log(D!)``.

    Parameters
    ----------
    sequence : list[float] | list[int]
        Input scalar sequence.
    D : int
        Embedding dimension.
    tau : int
        Embedding delay.
    epsilon : float
        Absolute tolerance for tie detection.

    Returns
    -------
    tuple[float | None, float, dict[int, int]]
        ``(pe, tie_rate, pattern_counts)`` where *pe* is the normalized
        permutation entropy (``None`` when no strict-order patterns exist),
        *tie_rate* is the fraction of windows excluded, and
        *pattern_counts* maps each Lehmer index to its occurrence count.
    """
    patterns, _tied_count, tie_rate = extract_ordinal_patterns(
        sequence, D=D, tau=tau, epsilon=epsilon
    )

    if not patterns:
        return None, tie_rate, {}

    counts = Counter(patterns)
    pattern_counts: dict[int, int] = dict(counts)

    # Shannon entropy: H = -sum(p * log(p))
    total = len(patterns)
    h = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            h -= p * math.log(p)

    # Normalize by log(D!) to map to [0, 1]
    h_max = math.log(FACTORIAL[D])
    if h_max < 1e-15:
        # D <= 1 has only one pattern, entropy is trivially 0
        return 0.0, tie_rate, pattern_counts

    pe = min(1.0, max(0.0, h / h_max))

    return pe, tie_rate, pattern_counts
