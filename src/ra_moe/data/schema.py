"""
Feature definitions and dataset schema for RA-MoE-4E.

This module centralizes column names used throughout the data pipeline.
Keeping feature definitions in one place makes the experimental inputs
explicit and avoids duplicating hard-coded column lists across scripts.
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Core columns required from the option dataset
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = (
    "secid",
    "date",
    "S",
    "K",
    "sigma",
    "market_price",
    "q",
    "r",
    "tau",
)


# ---------------------------------------------------------------------
# Raw column aliases
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "strike_price": "K",
    "impl_vol": "sigma",
    "div_yield": "q",
    "mid_price": "market_price",
}


# ---------------------------------------------------------------------
# Static features
# Expert 3: nonlinear cross-sectional MLP
# ---------------------------------------------------------------------

STATIC_FEATURES = (
    "S",
    "K",
    "moneyness",
    "tau",
    "sigma",
    "RV5",
    "RV20",
    "IVstd",
    "spread",
    "volume",
    "open_interest",
)


# ---------------------------------------------------------------------
# Residual features
# Expert 2: residual-correction MLP
#
# IMPORTANT:
# These reproduce the supplied final implementation.
# The thesis additionally describes Delta_BS as an input feature,
# but it is not explicitly included in the final supplied code.
# ---------------------------------------------------------------------

RESIDUAL_FEATURES = (
    "moneyness",
    "delta",
    "gamma",
    "vega",
    "theta",
    "skew",
    "spread",
)


# ---------------------------------------------------------------------
# Temporal features
# Expert 4: Transformer
#
# IMPORTANT:
# The supplied final implementation uses open_interest.
# The thesis methodology instead describes VIX as the fifth temporal
# feature. This discrepancy is intentionally preserved and documented.
# ---------------------------------------------------------------------

TEMPORAL_FEATURES = (
    "sigma",
    "volume",
    "ret",
    "spread",
    "open_interest",
)


# ---------------------------------------------------------------------
# Additional regime/state features for the gating network
#
# In the supplied implementation these features are concatenated with
# the standardized static feature representation.
# ---------------------------------------------------------------------

GATE_EXTRA_FEATURES = (
    "sigma",
    "moneyness",
    "tau",
    "RV5",
    "RV20",
    "dIV",
    "spread",
    "OI_chg",
)
