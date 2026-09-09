"""
Data-loading pipeline for RA-MoE-4E.

This module refactors the data preparation workflow from the supplied
final experimental implementation.

The pipeline connects:
- CSV loading and basic preprocessing
- temporal feature construction
- optional observation sampling
- Black-Scholes-Merton baseline calculation
- static, residual, and gating feature construction

Train/validation/test splitting and PyTorch dataset construction remain
separate from this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.bsm import bsm_price_numpy
from .features import (
    add_gate_features,
    build_feature_matrices,
)
from .preprocessing import preprocess_basic
from .temporal import (
    DEFAULT_SEQUENCE_LENGTH,
    build_temporal_features,
)


def load_data(
    data_path: str,
    sample_fraction: float = 1.0,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    temporal_group_key: str = "optionid",
    seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Load and prepare data for RA-MoE-4E.

    Parameters
    ----------
    data_path
        Path to the cleaned option CSV file.

    sample_fraction
        Fraction of aligned temporal observations retained.
        The original full experiment used 1.0.

    sequence_length
        Temporal lookback length. The supplied final implementation
        used 10.

    temporal_group_key
        Grouping key used to construct temporal histories.
        The supplied final implementation uses ``optionid`` when
        available.

    seed
        Random seed used when observation sampling is enabled.

    Returns
    -------
    static_features
        Static feature matrix.

    temporal_features
        Fixed-length temporal feature windows.

    targets
        Observed market option prices.

    bsm_prices
        Black-Scholes-Merton baseline prices.

    residual_extra_features
        Additional features for the residual-correction expert.

    gate_extra_features
        Additional features for the gating network.

    aligned_df
        DataFrame aligned row-by-row with all returned arrays.
    """
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(
            "sample_fraction must satisfy 0 < sample_fraction <= 1."
        )

    # -----------------------------------------------------------------
    # Load and basic preprocessing
    # -----------------------------------------------------------------

    df = pd.read_csv(data_path)

    df = preprocess_basic(df)

    # -----------------------------------------------------------------
    # Temporal features
    # -----------------------------------------------------------------

    temporal_features, aligned_df = build_temporal_features(
        df=df,
        group_key=temporal_group_key,
        sequence_length=sequence_length,
    )

    # -----------------------------------------------------------------
    # Optional sampling
    #
    # The supplied implementation samples only after temporal windows
    # have been constructed so that sequence construction itself is not
    # disrupted.
    # -----------------------------------------------------------------

    if sample_fraction < 1.0:
        n_observations = len(aligned_df)

        sample_size = int(
            n_observations * sample_fraction
        )

        np.random.seed(seed)

        sampled_indices = np.random.choice(
            n_observations,
            sample_size,
            replace=False,
        )

        aligned_df = (
            aligned_df
            .iloc[sampled_indices]
            .reset_index(drop=True)
        )

        temporal_features = temporal_features[
            sampled_indices
        ]

    # -----------------------------------------------------------------
    # Black-Scholes-Merton baseline
    # -----------------------------------------------------------------

    bsm_prices = bsm_price_numpy(
        spot=aligned_df["S"].to_numpy(),
        strike=aligned_df["K"].to_numpy(),
        rate=aligned_df["r"].to_numpy(),
        dividend_yield=aligned_df["q"].to_numpy(),
        volatility=aligned_df["sigma"].to_numpy(),
        time_to_maturity=aligned_df["tau"].to_numpy(),
        option_type="call",
    ).astype(np.float32)

    # The original implementation first calculates call prices for all
    # rows, then replaces put-option rows with BSM put prices.
    if "cp_flag" in aligned_df.columns:
        is_put = (
            aligned_df["cp_flag"]
            .astype(str)
            .str.upper()
            .str.startswith("P")
            .to_numpy()
        )

        if is_put.any():
            bsm_prices[is_put] = bsm_price_numpy(
                spot=aligned_df.loc[is_put, "S"].to_numpy(),
                strike=aligned_df.loc[is_put, "K"].to_numpy(),
                rate=aligned_df.loc[is_put, "r"].to_numpy(),
                dividend_yield=aligned_df.loc[is_put, "q"].to_numpy(),
                volatility=aligned_df.loc[is_put, "sigma"].to_numpy(),
                time_to_maturity=aligned_df.loc[is_put, "tau"].to_numpy(),
                option_type="put",
            ).astype(np.float32)

    # -----------------------------------------------------------------
    # Gate-specific derived features
    # -----------------------------------------------------------------

    aligned_df = add_gate_features(
        aligned_df,
        group_key="secid",
    )

    # -----------------------------------------------------------------
    # Feature matrices
    # -----------------------------------------------------------------

    feature_matrices = build_feature_matrices(
        aligned_df
    )

    static_features = feature_matrices["static"]
    residual_extra_features = feature_matrices["residual"]
    gate_extra_features = feature_matrices["gate_extra"]

    targets = aligned_df[
        "market_price"
    ].to_numpy(dtype=np.float32)

    return (
        static_features,
        temporal_features,
        targets,
        bsm_prices,
        residual_extra_features,
        gate_extra_features,
        aligned_df,
    )
