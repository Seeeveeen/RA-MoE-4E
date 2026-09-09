"""
Temporal feature construction for RA-MoE-4E.

This module refactors the temporal preprocessing logic from the
supplied final experimental implementation.

Important
---------
The original implementation constructs rolling statistics and
fixed-length temporal windows within a grouping key, typically
``optionid`` when available.

This behavior is preserved for reproducibility. It differs from the
thesis methodology description, which describes market-level daily
history shared across option observations. That discrepancy is
documented separately and is not silently corrected here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import TEMPORAL_FEATURES


DEFAULT_SEQUENCE_LENGTH = 10


def add_return_and_rolling_features(
    df: pd.DataFrame,
    group_key: str,
) -> pd.DataFrame:
    """
    Construct return and rolling volatility features.

    Features
    --------
    ret
        Log return of the underlying price within each group.

    RV5
        Rolling 5-observation standard deviation of returns.

    RV20
        Rolling 20-observation standard deviation of returns.

    IVstd
        Rolling 20-observation standard deviation of implied
        volatility.

    Notes
    -----
    This function preserves the grouping logic of the supplied final
    implementation. Rolling windows therefore refer to observations
    within ``group_key`` rather than being redefined as a market-level
    daily series.
    """
    if group_key not in df.columns:
        raise ValueError(
            f"Grouping column {group_key!r} is not present in the dataset."
        )

    df = df.copy()

    df = df.sort_values(
        [group_key, "date"]
    ).reset_index(drop=True)

    safe_spot = df["S"].where(
        df["S"] > 0
    )

    df["logS"] = np.log(safe_spot)

    df["ret"] = (
        df.groupby(group_key, sort=False)["logS"]
        .diff()
        .fillna(0.0)
    )

    grouped_returns = df.groupby(
        group_key,
        sort=False,
    )["ret"]

    df["RV5"] = grouped_returns.transform(
        lambda values: (
            values
            .rolling(5, min_periods=1)
            .std()
            .fillna(0.0)
        )
    )

    df["RV20"] = grouped_returns.transform(
        lambda values: (
            values
            .rolling(20, min_periods=1)
            .std()
            .fillna(0.0)
        )
    )

    df["IVstd"] = (
        df.groupby(
            group_key,
            sort=False,
        )["sigma"]
        .transform(
            lambda values: (
                values
                .rolling(20, min_periods=1)
                .std()
                .fillna(0.0)
            )
        )
    )

    return df


def build_temporal_windows(
    df: pd.DataFrame,
    group_key: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Construct fixed-length temporal windows.

    Parameters
    ----------
    df
        Dataset containing temporal features.

    group_key
        Column defining independent temporal groups.

    sequence_length
        Number of observations in each lookback window.

    Returns
    -------
    temporal_windows
        NumPy array with shape:

        ``(n_observations, sequence_length, n_temporal_features)``

    aligned_df
        DataFrame ordered consistently with ``temporal_windows``.

    Notes
    -----
    Short histories are left-padded using the earliest available
    observation, matching the behavior of the supplied final code.
    """
    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be greater than zero."
        )

    if group_key not in df.columns:
        raise ValueError(
            f"Grouping column {group_key!r} is not present in the dataset."
        )

    missing_features = [
        feature
        for feature in TEMPORAL_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing temporal features: "
            + ", ".join(missing_features)
        )

    aligned_df = (
        df.sort_values([group_key, "date"])
        .reset_index(drop=True)
        .copy()
    )

    windows: list[np.ndarray] = []

    for _, group in aligned_df.groupby(
        group_key,
        sort=False,
    ):
        feature_matrix = group.loc[
            :,
            TEMPORAL_FEATURES,
        ].to_numpy(dtype=np.float32)

        for position in range(len(group)):
            start = max(
                0,
                position - sequence_length + 1,
            )

            window = feature_matrix[
                start : position + 1
            ]

            if len(window) < sequence_length:
                padding_length = (
                    sequence_length - len(window)
                )

                first_observation = window[0:1]

                padding = np.repeat(
                    first_observation,
                    padding_length,
                    axis=0,
                )

                window = np.concatenate(
                    [padding, window],
                    axis=0,
                )

            windows.append(window)

    temporal_windows = np.stack(
        windows,
        axis=0,
    ).astype(np.float32)

    return temporal_windows, aligned_df


def build_temporal_features(
    df: pd.DataFrame,
    group_key: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Run the temporal feature pipeline.

    This function:

    1. sorts observations within the requested grouping key;
    2. constructs returns and rolling statistics;
    3. creates fixed-length temporal windows;
    4. returns windows together with the aligned DataFrame.
    """
    featured_df = add_return_and_rolling_features(
        df=df,
        group_key=group_key,
    )

    return build_temporal_windows(
        df=featured_df,
        group_key=group_key,
        sequence_length=sequence_length,
    )
