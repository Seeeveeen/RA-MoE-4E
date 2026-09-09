"""Core evaluation utilities for RA-MoE-4E."""

from .metrics import (
    baseline_comparison,
    compute_bsm_metrics,
    compute_metrics,
    error_stats,
    gating_analysis,
)
from .statistical import dm_test

__all__ = [
    "compute_metrics",
    "compute_bsm_metrics",
    "baseline_comparison",
    "gating_analysis",
    "error_stats",
    "dm_test",
]
