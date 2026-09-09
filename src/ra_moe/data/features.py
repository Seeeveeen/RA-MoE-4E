"""
Feature construction for RA-MoE-4E.

This module constructs the feature matrices used by the specialized
experts and the regime-aware gating network.

Feature construction is intentionally separated from scaling.
Normalization parameters must be fitted on the training set only and
are handled later in the data pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    GATE_EXTRA_FEATURES,
    RESIDUAL_FEATURES,
    STATIC_FEATURES,
)


def add_gate_features(
    df: pd.DataFrame,
    group_key: str = "secid",
) -> pd.DataFrame:
    """
    Construct additional regime/state features used by the gate.

    The input row order is preserved so that the resulting DataFrame
    remains aligned with the previously constructed temporal windows.
    """
    required = {
        group_key,
        "sigma",
        "open_interest",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing columns required for gate features: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["IVmean20"] = (
        df.groupby(
            group_key,
            sort=False,
        )["sigma"]
        .transform(
            lambda values: values.rolling(
                20,
                min_periods=1,
            ).mean()
        )
    )

    df["dIV"] = (
        df["sigma"] - df["IVmean20"]
    )

    df["OI_chg"] = (
        df.groupby(
            group_key,
            sort=False,
        )["open_interest"]
        .diff()
        .fillna(0.0)
    )

    return df


def _validate_feature_columns(
    df: pd.DataFrame,
    feature_names: tuple[str, ...],
    feature_group: str,
) -> None:
    """
    Validate that all requested feature columns exist.
    """
    missing = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {feature_group} features: "
            + ", ".join(missing)
        )


def build_static_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the feature matrix for Expert 3.

    Expert 3 is the static nonlinear MLP.

    Returns
    -------
    numpy.ndarray
        Matrix with shape
        ``(n_observations, n_static_features)``.
    """
    _validate_feature_columns(
        df=df,
        feature_names=STATIC_FEATURES,
        feature_group="static",
    )

    return df.loc[
        :,
        STATIC_FEATURES,
    ].to_numpy(dtype=np.float32)


def build_residual_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the feature matrix for Expert 2.

    Expert 2 is the residual-correction MLP.

    The supplied final implementation uses a zero-valued skew
    placeholder when no skew column is available.
    """
    df = df.copy()

    if "skew" not in df.columns:
        df["skew"] = 0.0

    _validate_feature_columns(
        df=df,
        feature_names=RESIDUAL_FEATURES,
        feature_group="residual",
    )

    return df.loc[
        :,
        RESIDUAL_FEATURES,
    ].to_numpy(dtype=np.float32)


def build_gate_extra_features(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Build the additional regime/state feature matrix for the gate.

    These features are later combined with the static representation
    to reproduce the gating input structure of the supplied final
    implementation.

    Scaling is deliberately not performed here.

    Returns
    -------
    numpy.ndarray
        Matrix with shape
        ``(n_observations, n_gate_extra_features)``.
    """
    _validate_feature_columns(
        df=df,
        feature_names=GATE_EXTRA_FEATURES,
        feature_group="gating",
    )

    return df.loc[
        :,
        GATE_EXTRA_FEATURES,
    ].to_numpy(dtype=np.float32)


def build_feature_matrices(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """
    Construct all non-temporal model feature matrices.

    Returns
    -------
    dict
        Dictionary containing:

        - ``static``
        - ``residual``
        - ``gate_extra``
    """
    return {
        "static": build_static_features(df),
        "residual": build_residual_features(df),
        "gate_extra": build_gate_extra_features(df),
    }
