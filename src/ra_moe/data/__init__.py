"""Data processing utilities for RA-MoE-4E."""

from .features import (
    add_gate_features,
    build_feature_matrices,
    build_gate_extra_features,
    build_residual_features,
    build_static_features,
)
from .preprocessing import (
    add_basic_features,
    normalize_columns,
    preprocess_basic,
    validate_required_columns,
)
from .schema import (
    GATE_EXTRA_FEATURES,
    RESIDUAL_FEATURES,
    STATIC_FEATURES,
    TEMPORAL_FEATURES,
)
from .temporal import (
    DEFAULT_SEQUENCE_LENGTH,
    add_return_and_rolling_features,
    build_temporal_features,
    build_temporal_windows,
)

__all__ = [
    "add_basic_features",
    "normalize_columns",
    "preprocess_basic",
    "validate_required_columns",
    "STATIC_FEATURES",
    "RESIDUAL_FEATURES",
    "TEMPORAL_FEATURES",
    "GATE_EXTRA_FEATURES",
    "DEFAULT_SEQUENCE_LENGTH",
    "add_return_and_rolling_features",
    "build_temporal_features",
    "build_temporal_windows",
    "add_gate_features",
    "build_feature_matrices",
    "build_gate_extra_features",
    "build_residual_features",
    "build_static_features",
]
