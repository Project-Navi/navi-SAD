"""Shared statistical utilities.

Pure functions for effect size computation. No domain-specific logic.
Both pilot and analysis modules import from here.
"""

from navi_sad.stats.effect_size import GuardedStat, compute_cohens_d

__all__ = [
    "GuardedStat",
    "compute_cohens_d",
]
